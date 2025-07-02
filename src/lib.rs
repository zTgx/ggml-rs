pub mod gguf;

pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFFFFFF;

pub const LLAMA_TOKEN_NULL: i32 = -1;

pub const LLAMA_FILE_MAGIC_GGLA: u32 = 0x67676c61; // 'ggla'
pub const LLAMA_FILE_MAGIC_GGSN: u32 = 0x6767736e; // 'ggsn'
pub const LLAMA_FILE_MAGIC_GGSQ: u32 = 0x67677371; // 'ggsq'

pub const LLAMA_SESSION_MAGIC: u32 = LLAMA_FILE_MAGIC_GGSN;
pub const LLAMA_SESSION_VERSION: u32 = 9;

pub const LLAMA_STATE_SEQ_MAGIC: u32 = LLAMA_FILE_MAGIC_GGSQ;
pub const LLAMA_STATE_SEQ_VERSION: u32 = 2;

// Type aliases
pub type LlamaPos = i32;
pub type LlamaToken = i32;
pub type LlamaSeqId = i32;

// Vocabulary types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaVocabType {
    None = 0, // For models without vocab
    Spm = 1,  // LLaMA tokenizer based on byte-level BPE with byte fallback
    Bpe = 2,  // GPT-2 tokenizer based on byte-level BPE
    Wpm = 3,  // BERT tokenizer based on WordPiece
    Ugm = 4,  // T5 tokenizer based on Unigram
    Rwkv = 5, // RWKV tokenizer based on greedy tokenization
}

// Pre-tokenization types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaVocabPreType {
    Default = 0,
    Llama3 = 1,
    DeepseekLlm = 2,
    DeepseekCoder = 3,
    Falcon = 4,
    Mpt = 5,
    Starcoder = 6,
    Gpt2 = 7,
    Refact = 8,
    CommandR = 9,
    Stablelm2 = 10,
    Qwen2 = 11,
    Olmo = 12,
    Dbrx = 13,
    Smaug = 14,
    Poro = 15,
    Chatglm3 = 16,
    Chatglm4 = 17,
    Viking = 18,
    Jais = 19,
    Tekken = 20,
    Smolllm = 21,
    Codeshell = 22,
    Bloom = 23,
    Gpt3Finnish = 24,
    Exaone = 25,
    Chameleon = 26,
    Minerva = 27,
    Deepseek3Llm = 28,
    Gpt4o = 29,
    Superbpe = 30,
    Trillion = 31,
    Bailingmoe = 32,
    Llama4 = 33,
    Pixtral = 34,
    SeedCoder = 35,
}

// Rope types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaRopeType {
    None = -1,
    Norm = 0,
    Neox = 1,   // GGML_ROPE_TYPE_NEOX
    Mrope = 2,  // GGML_ROPE_TYPE_MROPE
    Vision = 3, // GGML_ROPE_TYPE_VISION
}

// Token types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaTokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

// Token attributes (bitflags)
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct LlamaTokenAttr: u32 {
        const UNDEFINED = 0;
        const UNKNOWN = 1 << 0;
        const UNUSED = 1 << 1;
        const NORMAL = 1 << 2;
        const CONTROL = 1 << 3;  // SPECIAL?
        const USER_DEFINED = 1 << 4;
        const BYTE = 1 << 5;
        const NORMALIZED = 1 << 6;
        const LSTRIP = 1 << 7;
        const RSTRIP = 1 << 8;
        const SINGLE_WORD = 1 << 9;
    }
}

// Model file types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaFtype {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ5_1 = 9,
    MostlyQ2K = 10,
    MostlyQ3KS = 11,
    MostlyQ3KM = 12,
    MostlyQ3KL = 13,
    MostlyQ4KS = 14,
    MostlyQ4KM = 15,
    MostlyQ5KS = 16,
    MostlyQ5KM = 17,
    MostlyQ6K = 18,
    MostlyIq2Xxs = 19,
    MostlyIq2Xs = 20,
    MostlyQ2KS = 21,
    MostlyIq3Xs = 22,
    MostlyIq3Xxs = 23,
    MostlyIq1S = 24,
    MostlyIq4Nl = 25,
    MostlyIq3S = 26,
    MostlyIq3M = 27,
    MostlyIq2S = 28,
    MostlyIq2M = 29,
    MostlyIq4Xs = 30,
    MostlyIq1M = 31,
    MostlyBf16 = 32,
    MostlyTq1_0 = 36,
    MostlyTq2_0 = 37,
    Guessed = 1024,
}

// Rope scaling types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaRopeScalingType {
    Unspecified = -1,
    None = 0,
    Linear = 1,
    Yarn = 2,
    Longrope = 3,
}
pub const LLAMA_ROPE_SCALING_TYPE_MAX_VALUE: i32 = 3;

// Pooling types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaPoolingType {
    Unspecified = -1,
    None = 0,
    Mean = 1,
    Cls = 2,
    Last = 3,
    Rank = 4,
}

// Attention types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaAttentionType {
    Unspecified = -1,
    Causal = 0,
    NonCausal = 1,
}

// Split modes
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaSplitMode {
    None = 0,
    Layer = 1,
    Row = 2,
}
