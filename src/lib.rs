use regex::Regex;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct GGUFComponents {
    pub base_name: Option<String>,
    pub size_label: Option<String>,
    pub fine_tune: Option<String>,
    pub version: String,
    pub encoding: Option<String>,
    pub type_field: Option<String>,
    pub shard: Option<String>,
}

pub fn parse_gguf_filename(filename: &str) -> Option<GGUFComponents> {
    let gguf_regex = Regex::new(
        r"^(?P<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?P<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?P<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?P<Version>v\d+(?:\.\d+)*))(?:-(?P<Encoding>[\w_]+))?(?:-(?P<Type>LoRA|vocab))?(?:-(?P<Shard>\d{5}-of-\d{5}))?\.gguf$"
    ).expect("Failed to compile regex");

    match gguf_regex.captures(filename) {
        Some(caps) => {
            let mut encoding = caps.name("Encoding").map(|m| m.as_str().to_string());
            let type_field = caps.name("Type").map(|m| m.as_str().to_string());

            // Post-process to handle LoRA and vocab exclusion from Encoding
            if let Some(enc) = encoding.clone() {
                if enc == "LoRA" || enc == "vocab" {
                    encoding = None;
                }
            }

            Some(GGUFComponents {
                base_name: caps.name("BaseName").map(|m| m.as_str().to_string()),
                size_label: caps.name("SizeLabel").map(|m| m.as_str().to_string()),
                fine_tune: caps.name("FineTune").map(|m| m.as_str().to_string()),
                version: caps
                    .name("Version")
                    .map_or("v1.0".to_string(), |m| m.as_str().to_string()),
                encoding,
                type_field,
                shard: caps.name("Shard").map(|m| m.as_str().to_string()),
            })
        }
        None => None,
    }
}

pub fn parse_gguf_filename_map(filename: &str) -> Option<HashMap<String, Option<String>>> {
    parse_gguf_filename(filename).map(|components| {
        let mut map = HashMap::new();
        map.insert("BaseName".to_string(), components.base_name);
        map.insert("SizeLabel".to_string(), components.size_label);
        map.insert("FineTune".to_string(), components.fine_tune);
        map.insert("Version".to_string(), Some(components.version));
        map.insert("Encoding".to_string(), components.encoding);
        map.insert("Type".to_string(), components.type_field);
        map.insert("Shard".to_string(), components.shard);
        map
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_gguf_filename() {
        let test_cases = vec![
            (
                "Mixtral-8x7B-v0.1-KQ2.gguf",
                json!({
                    "BaseName": "Mixtral",
                    "SizeLabel": "8x7B",
                    "FineTune": null,
                    "Version": "v0.1",
                    "Encoding": "KQ2",
                    "Type": null,
                    "Shard": null
                }),
            ),
            (
                "Grok-100B-v1.0-Q4_0-00003-of-00009.gguf",
                json!({
                    "BaseName": "Grok",
                    "SizeLabel": "100B",
                    "FineTune": null,
                    "Version": "v1.0",
                    "Encoding": "Q4_0",
                    "Type": null,
                    "Shard": "00003-of-00009"
                }),
            ),
            (
                "Hermes-2-Pro-Llama-3-8B-v1.0-F16.gguf",
                json!({
                    "BaseName": "Hermes-2-Pro-Llama-3",
                    "SizeLabel": "8B",
                    "FineTune": null,
                    "Version": "v1.0",
                    "Encoding": "F16",
                    "Type": null,
                    "Shard": null
                }),
            ),
            (
                "Phi-3-mini-3.8B-ContextLength4k-instruct-v1.0.gguf",
                json!({
                    "BaseName": "Phi-3-mini",
                    "SizeLabel": "3.8B-ContextLength4k",
                    "FineTune": "instruct",
                    "Version": "v1.0",
                    "Encoding": null,
                    "Type": null,
                    "Shard": null
                }),
            ),
            ("not-a-known-arrangement.gguf", json!(null)),
        ];

        for (filename, expected) in test_cases {
            let result = parse_gguf_filename_map(filename);
            let result_json = json!(result);
            assert_eq!(
                result_json, expected,
                "Failed for filename: {}. Expected: {:?}, Got: {:?}",
                filename, expected, result_json
            );
        }
    }
}