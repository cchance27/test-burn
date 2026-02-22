use rayon::prelude::*;
use unicode_normalization::UnicodeNormalization;

use super::BPETokenizer;
use crate::MetalError;

impl BPETokenizer {
    /// Encode text into tokens (defaults to encode_simd now that we've benchmarked it as fastest in all lengths).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        self.encode_simd(text)
    }

    fn process_non_special_subtext(
        &self,
        subtext: &str,
        processor: &mut impl FnMut(&str) -> Result<(), MetalError>,
    ) -> Result<(), MetalError> {
        if matches!(self.pre_tokenizer_kind, super::pretokenizer::PreTokenizerKind::Llama3Like) {
            return self.process_llama3_like_pieces(subtext, processor);
        }

        for submat in self.token_piece_re.find_iter(subtext) {
            processor(submat?.as_str())?;
        }
        Ok(())
    }

    fn process_llama3_like_pieces(
        &self,
        subtext: &str,
        processor: &mut impl FnMut(&str) -> Result<(), MetalError>,
    ) -> Result<(), MetalError> {
        if subtext.is_empty() {
            return Ok(());
        }

        let chars: Vec<char> = subtext.chars().collect();
        let mut char_starts: Vec<usize> = subtext.char_indices().map(|(idx, _)| idx).collect();
        char_starts.push(subtext.len());

        let mut emit_piece = |start_char: usize, end_char: usize| -> Result<(), MetalError> {
            if end_char <= start_char {
                return Ok(());
            }
            let start_byte = char_starts[start_char];
            let end_byte = char_starts[end_char];
            processor(&subtext[start_byte..end_byte])
        };

        let n = chars.len();
        let mut pos = 0usize;
        while pos < n {
            let start = pos;
            let c = chars[pos];

            if c == '\'' && pos + 1 < n {
                let c1 = chars[pos + 1].to_ascii_lowercase();
                if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
                    pos += 2;
                    emit_piece(start, pos)?;
                    continue;
                }
                if pos + 2 < n {
                    let c2 = chars[pos + 2].to_ascii_lowercase();
                    if ((c1 == 'v' || c1 == 'r') && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
                        pos += 3;
                        emit_piece(start, pos)?;
                        continue;
                    }
                }
            }

            if c != '\r' && c != '\n' && !c.is_numeric() && (c.is_alphabetic() || (pos + 1 < n && chars[pos + 1].is_alphabetic())) {
                pos += 1;
                while pos < n && chars[pos].is_alphabetic() {
                    pos += 1;
                }
                emit_piece(start, pos)?;
                continue;
            }

            if c.is_numeric() {
                let mut group_start = pos;
                while pos < n && chars[pos].is_numeric() {
                    pos += 1;
                    if pos - group_start >= 3 {
                        emit_piece(group_start, pos)?;
                        group_start = pos;
                    }
                }
                emit_piece(group_start, pos)?;
                continue;
            }

            let mut test_pos = pos;
            if c == ' ' && pos + 1 < n {
                test_pos = pos + 1;
            }
            let punct_like =
                (test_pos < n) && !chars[test_pos].is_whitespace() && !chars[test_pos].is_alphabetic() && !chars[test_pos].is_numeric();
            if punct_like {
                if c == ' ' {
                    pos += 1;
                }
                while pos < n {
                    let ch = chars[pos];
                    if ch.is_whitespace() || ch.is_alphabetic() || ch.is_numeric() {
                        break;
                    }
                    pos += 1;
                }
                while pos < n && (chars[pos] == '\r' || chars[pos] == '\n') {
                    pos += 1;
                }
                emit_piece(start, pos)?;
                continue;
            }

            let mut num_whitespace = 0usize;
            let mut last_end_r_or_n = 0usize;
            while pos + num_whitespace < n && chars[pos + num_whitespace].is_whitespace() {
                let ch = chars[pos + num_whitespace];
                if ch == '\r' || ch == '\n' {
                    last_end_r_or_n = pos + num_whitespace + 1;
                }
                num_whitespace += 1;
            }

            if last_end_r_or_n > 0 {
                pos = last_end_r_or_n;
                emit_piece(start, pos)?;
                continue;
            }
            if num_whitespace > 1 && pos + num_whitespace < n {
                pos += num_whitespace - 1;
                emit_piece(start, pos)?;
                continue;
            }
            if num_whitespace > 0 {
                pos += num_whitespace;
                emit_piece(start, pos)?;
                continue;
            }

            pos += 1;
            emit_piece(start, pos)?;
        }

        Ok(())
    }

    fn process_pieces(&self, text: &str, mut processor: impl FnMut(&str) -> Result<(), MetalError>) -> Result<(), MetalError> {
        let mut start = 0usize;
        let mut pos = 0usize;

        while pos < text.len() {
            let rest = &text[pos..];
            let mut matched_special: Option<&str> = None;
            if rest.as_bytes().first().copied() == Some(b'<') {
                for tok in &self.special_literal_tokens {
                    let tok = tok.as_ref();
                    if rest.starts_with(tok) {
                        matched_special = Some(tok);
                        break;
                    }
                }
                if matched_special.is_none()
                    && let Some(mat) = self.special_token_re.find_iter(rest).next()
                {
                    let mat = mat?;
                    if mat.start() == 0 {
                        matched_special = Some(mat.as_str());
                    }
                }
            }

            if let Some(tok) = matched_special {
                if pos > start {
                    let subtext = &text[start..pos];
                    self.process_non_special_subtext(subtext, &mut processor)?;
                }
                processor(tok)?;
                pos += tok.len();
                start = pos;
                continue;
            }

            let ch_len = rest.chars().next().map(char::len_utf8).unwrap_or(1);
            pos += ch_len;
        }

        if start < text.len() {
            let subtext = &text[start..];
            self.process_non_special_subtext(subtext, &mut processor)?;
        }
        Ok(())
    }

    /// Encode text into tokens using the serial implementation.
    pub fn encode_serial(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        self.process_pieces(&norm_text, |piece| {
            self.bpe_encode(piece, &mut token_ids);
            Ok(())
        })?;

        Ok(token_ids)
    }

    /// Encode text into tokens using the SIMD-optimized (ID-based) implementation.
    pub fn encode_simd(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        self.process_pieces(&norm_text, |piece| {
            self.bpe_encode_ids(piece, &mut token_ids);
            Ok(())
        })?;

        Ok(token_ids)
    }

    /// Encode text into tokens using the parallel implementation.
    pub fn encode_parallel(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        // In the parallel version, we must collect the pieces first before processing in parallel.
        // The streaming approach is not directly applicable with rayon's `par_iter`.
        let mut pieces = Vec::new();
        self.process_pieces(&norm_text, |piece| {
            pieces.push(piece.to_string());
            Ok(())
        })?;

        let tokens_from_pieces: Vec<Vec<u32>> = pieces
            .par_iter()
            .map(|piece| {
                let mut local_token_ids = Vec::new();
                self.bpe_encode_ids(piece, &mut local_token_ids);
                local_token_ids
            })
            .collect();

        token_ids.extend(tokens_from_pieces.into_iter().flatten());

        Ok(token_ids)
    }

    /// Encode text into tokens using the parallel SIMD-optimized (ID-based) implementation.
    pub fn encode_simd_parallel(&self, text: &str) -> Result<Vec<u32>, MetalError> {
        let mut norm_text = String::with_capacity(text.len());
        text.nfc().for_each(|c| norm_text.push(c));

        let mut token_ids = Vec::with_capacity(norm_text.len() / 2);
        if let Some(bos_id) = self.special_tokens.bos_token_id
            && self.add_bos_token
        {
            token_ids.push(bos_id);
        }

        let mut pieces = Vec::new();
        self.process_pieces(&norm_text, |piece| {
            pieces.push(piece.to_string());
            Ok(())
        })?;

        let tokens_from_pieces: Vec<Vec<u32>> = pieces
            .par_iter()
            .map(|piece| {
                let mut local_token_ids = Vec::new();
                self.bpe_encode_ids(piece, &mut local_token_ids);
                local_token_ids
            })
            .collect();

        token_ids.extend(tokens_from_pieces.into_iter().flatten());

        Ok(token_ids)
    }

    fn bpe_encode(&self, text: &str, token_ids: &mut Vec<u32>) {
        if let Some(id) = self.vocab_r.get(text) {
            token_ids.push(*id);
            return;
        }

        let mut token_unicode = String::with_capacity(text.len());
        for &b in text.as_bytes() {
            token_unicode.push(self.byte_encoder_array[b as usize]);
        }

        if self.ignore_merges {
            if let Some(&id) = self.vocab_r.get(&token_unicode) {
                token_ids.push(id);
                return;
            }
            for ch in token_unicode.chars() {
                let mut utf8 = [0_u8; 4];
                let encoded = ch.encode_utf8(&mut utf8);
                if let Some(&id) = self.vocab_r.get(encoded) {
                    token_ids.push(id);
                } else {
                    token_ids.push(0);
                }
            }
            return;
        }

        let mut pieces: Vec<String> = token_unicode.chars().map(|c| c.to_string()).collect();
        if pieces.is_empty() {
            return;
        }
        loop {
            let mut min_rank = u32::MAX;
            let mut merge_pos = None;

            for i in 0..pieces.len().saturating_sub(1) {
                let pair_key = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair_key)
                    && rank < min_rank
                {
                    min_rank = rank;
                    merge_pos = Some(i);
                }
            }

            if let Some(pos) = merge_pos {
                let merged = format!("{}{}", pieces[pos], pieces[pos + 1]);
                pieces.splice(pos..pos + 2, std::iter::once(merged));
            } else {
                break;
            }
        }
        for piece in pieces {
            if let Some(&id) = self.vocab_r.get(&piece) {
                token_ids.push(id);
            } else {
                token_ids.push(0); // UNK
            }
        }
    }

    fn bpe_encode_ids(&self, text: &str, token_ids: &mut Vec<u32>) {
        if let Some(id) = self.vocab_r.get(text) {
            token_ids.push(*id);
            return;
        }

        let mut token_unicode = String::with_capacity(text.len());
        for &b in text.as_bytes() {
            token_unicode.push(self.byte_encoder_array[b as usize]);
        }

        if self.ignore_merges {
            if let Some(&id) = self.vocab_r.get(&token_unicode) {
                token_ids.push(id);
                return;
            }
            token_ids.extend(token_unicode.chars().map(|c| *self.char_vocab.get(&c).unwrap_or(&0)));
            return;
        }

        let mut piece_ids: Vec<u32> = Vec::with_capacity(token_unicode.len());
        piece_ids.extend(
            token_unicode.chars().map(|c| *self.char_vocab.get(&c).unwrap_or(&0)), // Use UNK for unknown chars
        );

        if piece_ids.is_empty() {
            return;
        }

        if piece_ids.len() == 1 {
            token_ids.extend(piece_ids);
            return;
        }

        loop {
            let mut min_rank = u32::MAX;
            let mut merge_pos = None;

            for i in 0..piece_ids.len().saturating_sub(1) {
                let pair_key = (piece_ids[i], piece_ids[i + 1]);
                if let Some(&rank) = self.merges_ranks.get(&pair_key)
                    && rank < min_rank
                {
                    min_rank = rank;
                    merge_pos = Some(i);
                }
            }

            if let Some(pos) = merge_pos {
                let pair_key = (piece_ids[pos], piece_ids[pos + 1]);
                let merged_id = self.merges_results[&pair_key];
                piece_ids.splice(pos..pos + 2, std::iter::once(merged_id));
            } else {
                break;
            }
        }
        token_ids.extend(piece_ids);
    }
}
