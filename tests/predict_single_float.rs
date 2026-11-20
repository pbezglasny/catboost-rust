#![cfg(all(catboost_text_count, catboost_embeddings))]

use catboost_rust::Model;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: each operation forwards the original pointers and layouts to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn load(name: &str) -> Model {
    Model::load(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

#[test]
fn first_and_repeated_predictions_match_python_without_allocating() {
    let before = ALLOCATIONS.get();
    std::hint::black_box(vec![0u8; 32]);
    assert!(ALLOCATIONS.get() > before);

    // Raw predictions from tests/fixtures/generate.py, rounded to two decimal places.
    for (name, expected) in [
        ("scalar.cbm", [&[1.28][..], &[2.72][..]]),
        ("multi.cbm", [&[1.28, 2.72][..], &[2.72, 1.28][..]]),
    ] {
        let model = load(name);
        let mut output = vec![f64::NAN; expected[0].len()];
        for i in [0, 1, 0] {
            let row = [[0.0], [3.0]][i];
            let before = ALLOCATIONS.get();
            model.predict_single_float_into(&row, &mut output).unwrap();
            assert_eq!(ALLOCATIONS.get(), before);
            for (actual, reference) in output.iter().zip(expected[i]) {
                assert!(
                    (actual - reference).abs() < 1e-12,
                    "{name}: {actual} vs {reference}"
                );
            }
        }
    }
}

#[test]
fn rejects_incorrect_lengths_and_recovers() {
    let model = load("scalar.cbm");
    let mut output = [42.0];
    for row in [&[][..], &[1.0, 2.0][..]] {
        assert!(model.predict_single_float_into(row, &mut output).is_err());
        assert_eq!(output, [42.0]);
    }
    for buffer in [&mut [][..], &mut [0.0; 2][..]] {
        assert!(model.predict_single_float_into(&[0.0], buffer).is_err());
    }
    model
        .predict_single_float_into(&[0.0], &mut output)
        .unwrap();
    assert!((output[0] - 1.28).abs() < 1e-12);
}

#[test]
fn rejects_non_numeric_models_before_writing_output() {
    for name in ["categorical.cbm", "text.cbm", "embedding.cbm"] {
        let model = load(name);
        let mut output = [42.0];
        let err = model
            .predict_single_float_into(&[], &mut output)
            .unwrap_err();
        assert!(err.description.contains("numeric-only"), "{name}: {err}");
        assert_eq!(output, [42.0]);
    }
}
