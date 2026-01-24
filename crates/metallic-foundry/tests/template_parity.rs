use metallic_foundry::template::{ChatTemplate, Message};

#[test]
fn test_qwen_template_parity() {
    let qwen_template = "{% for message in messages %}{% if loop.first and message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% endif %}{% if message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% if loop.last and add_generation_prompt %}<|im_start|>assistant\n{% endif %}{% endfor %}";
    let template = ChatTemplate::new(qwen_template);

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        },
    ];

    let rendered = template.render(&messages, None, None, true).unwrap();

    let expected =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(rendered, expected);
}

#[test]
fn test_llama3_template() {
    let llama3_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% if message['role'] == 'system' %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}{% endfor %}";
    let template = ChatTemplate::new(llama3_template);

    let messages = vec![Message {
        role: "user".to_string(),
        content: "Tell me a joke.".to_string(),
    }];

    let rendered = template.render(&messages, None, None, true).unwrap();
    let expected = "<|start_header_id|>user<|end_header_id|>\n\nTell me a joke.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    assert_eq!(rendered, expected);
}

#[test]
fn test_mistral_template() {
    let mistral_template = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}";

    let template = ChatTemplate::new(mistral_template);

    let messages = vec![Message {
        role: "user".to_string(),
        content: "What is 2+2?".to_string(),
    }];

    let rendered = template.render(&messages, Some("<s>"), Some("</s>"), true).unwrap();
    let expected = "<s>[INST] What is 2+2? [/INST]";
    assert_eq!(rendered, expected);
}
