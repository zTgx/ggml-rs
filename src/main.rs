use clap::Parser;
use ggml_rs::gguf::parse_gguf_filename;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Parse GGUF filenames into their components",
    long_about = "A command-line tool to parse GGUF filenames and output their components as JSON."
)]
struct Args {
    /// The GGUF filename to parse
    filename: String,
}

fn main() {
    let args = Args::parse();
    match parse_gguf_filename(&args.filename) {
        Some(components) => {
            let json_output = json!(components);
            println!("{}", serde_json::to_string_pretty(&json_output).unwrap());
        }
        None => {
            eprintln!("Error: Invalid GGUF filename: {}", args.filename);
            std::process::exit(1);
        }
    }
}
