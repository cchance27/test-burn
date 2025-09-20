use super::*;
use crate::metallic::{Context, Tensor};

#[test]
fn zeros_and_ones() {
    let ctx = Context::new().unwrap();
    let t0 = Tensor::zeros(vec![2, 3, 4], &ctx).unwrap();
    assert!(t0.as_slice().iter().all(|&x| x == 0.0));
    let t1 = Tensor::ones(vec![2, 3, 4], &ctx).unwrap();
    assert!(t1.as_slice().iter().all(|&x| x == 1.0));
}

#[test]
fn zeros_like_and_ones_like() {
    let ctx = Context::new().unwrap();
    let base = Tensor::create_tensor_from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], &ctx).unwrap();
    let z = base.zeros_like(&ctx).unwrap();
    assert_eq!(z.dims(), base.dims());
    assert!(z.as_slice().iter().all(|&x| x == 0.0));
    let o = base.ones_like(&ctx).unwrap();
    assert_eq!(o.dims(), base.dims());
    assert!(o.as_slice().iter().all(|&x| x == 1.0));
}

#[test]
fn elementwise_ops_and_fill() {
    let ctx = Context::new().unwrap();
    let a = Tensor::create_tensor_from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], &ctx).unwrap();
    let b = Tensor::create_tensor_from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2], &ctx).unwrap();
    let c = &a + &b;
    assert_eq!(c.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    let d = &a * &b;
    assert_eq!(d.as_slice(), &[5.0, 12.0, 21.0, 32.0]);
    let mut e = a.add_scalar(10.0).unwrap();
    assert_eq!(e.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
    e.fill(2.5);
    assert!(e.as_slice().iter().all(|&x| (x - 2.5).abs() < 1e-12));
}

#[test]
fn get_batch_and_from_existing_buffer() {
    let ctx = Context::new().unwrap();
    // base tensor: shape [2,3,4], values 0..24
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let mut base = Tensor::create_tensor(24, vec![2, 3, 4], &ctx).unwrap();
    base.as_mut_slice().copy_from_slice(&data);

    // get second batch
    let b1 = base.get_batch(1).unwrap();
    assert_eq!(b1.dims(), &[3, 4]);
    // expected slice
    let expected: Vec<f32> = data[12..].to_vec();
    assert_eq!(b1.as_slice(), expected.as_slice());

    // wrap the second batch region using from_existing_buffer
    let offset_bytes = 12 * std::mem::size_of::<f32>();
    let view =
        Tensor::from_existing_buffer(base.buf.clone(), vec![3, 4], &base.device, offset_bytes)
            .unwrap();
    assert_eq!(view.as_slice(), expected.as_slice());
}

#[test]
fn arange_helper() {
    let ctx = Context::new().unwrap();
    let t = Tensor::arange(6, vec![2, 3], &ctx).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn from_vec_helper() {
    let ctx = Context::new().unwrap();
    let v = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let t = Tensor::from_vec(v.clone(), vec![2, 3], &ctx).unwrap();
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.to_vec(), v);
}
