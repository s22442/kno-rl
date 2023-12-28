use std::{backtrace, panic};

pub fn init() {
    panic::set_hook(Box::new(|info| {
        let backtrace = backtrace::Backtrace::force_capture();
        println!("{info}\n");
        println!("{backtrace}");
    }));
}
