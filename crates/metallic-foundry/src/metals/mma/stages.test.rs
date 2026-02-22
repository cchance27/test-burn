#![cfg(test)]

use super::*;

#[test]
fn test_tile_config_auto_select() {
    assert_eq!(TileConfig::auto_select(1, 4096), TileConfig::SkinnyM);
    assert_eq!(TileConfig::auto_select(512, 4096), TileConfig::HighPerformance);
    assert_eq!(TileConfig::auto_select(512, 8), TileConfig::SkinnyN);
}

#[test]
fn test_tile_config_sizes() {
    let (bm, bn, bk, wm, wn) = TileConfig::Default.tile_sizes();
    assert_eq!((bm, bn, bk, wm, wn), (32, 32, 16, 2, 2));

    let tgp = TileConfig::Default.threads_per_tg();
    assert_eq!(tgp, 128); // 2*2*32
}
