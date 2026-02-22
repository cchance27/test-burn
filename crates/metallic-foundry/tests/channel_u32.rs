use metallic_foundry::{
    Foundry, workflow::{ChannelU32, ChannelU32Reader}
};

#[test]
fn channel_u32_basic_order_no_wrap() {
    let mut foundry = Foundry::new().expect("foundry init");
    let chan = ChannelU32::allocate(&mut foundry, 32).expect("alloc channel");

    foundry.start_capture().expect("capture");
    for v in 1u32..=16u32 {
        chan.push_scalar(&mut foundry, v).expect("push");
    }
    let cmd = foundry.end_capture().expect("end capture");
    cmd.wait_until_completed();

    let mut reader = ChannelU32Reader::new(chan);
    let mut got = Vec::new();
    reader.drain_into(&mut got).expect("drain");

    assert_eq!(got, (1u32..=16u32).collect::<Vec<_>>());
}

#[test]
fn channel_u32_wrap_drops_oldest() {
    let mut foundry = Foundry::new().expect("foundry init");
    let chan = ChannelU32::allocate(&mut foundry, 4).expect("alloc channel");

    foundry.start_capture().expect("capture");
    for v in 1u32..=10u32 {
        chan.push_scalar(&mut foundry, v).expect("push");
    }
    let cmd = foundry.end_capture().expect("end capture");
    cmd.wait_until_completed();

    let mut reader = ChannelU32Reader::new(chan);
    let mut got = Vec::new();
    reader.drain_into(&mut got).expect("drain");

    assert_eq!(got, vec![7, 8, 9, 10]);
}
