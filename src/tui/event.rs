use crossterm::event::{self, Event as CrosstermEvent, KeyEvent, MouseEvent};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum Event {
    Tick,
    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),
}

pub fn run_event_loop(sender: mpsc::Sender<Event>, tick_rate: Duration) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let receiver = sender.clone();
        loop {
            // poll for tick rate duration, if no event, sent tick event.
            if event::poll(tick_rate).expect("Failed to poll for event") {
                match event::read().expect("Failed to read event") {
                    CrosstermEvent::Key(key_event) => {
                        if receiver.send(Event::Key(key_event)).is_err() {
                            break;
                        }
                    }
                    CrosstermEvent::Mouse(mouse_event) => {
                        if receiver.send(Event::Mouse(mouse_event)).is_err() {
                            break;
                        }
                    }
                    CrosstermEvent::Resize(width, height) => {
                        if receiver.send(Event::Resize(width, height)).is_err() {
                            break;
                        }
                    }
                    CrosstermEvent::FocusGained | CrosstermEvent::FocusLost | CrosstermEvent::Paste(_) => {
                        // Ignore these events for now
                    }
                }
            } else if sender.send(Event::Tick).is_err() {
                break;
            }
        }
    })
}
