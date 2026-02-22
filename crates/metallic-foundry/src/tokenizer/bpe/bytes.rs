use rustc_hash::FxHashMap;

pub(super) fn bytes_to_unicode() -> FxHashMap<u8, char> {
    let mut bs = (b'!'..=b'~').chain(b'\xa1'..=b'\xac').chain(b'\xae'..=b'\xff').collect::<Vec<_>>();
    let mut cs = bs.iter().map(|b| *b as u32).collect::<Vec<_>>();
    let mut n = 0;
    for b in 0..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs.into_iter().map(|c| std::char::from_u32(c).unwrap()))
        .collect()
}
