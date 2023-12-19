package model.config

data class GPTConfig(
    val blockSize: Int = 1024,
    val vocabSize: Int = 50304, // GPT-2 vocab size is 50257, padded up to nearest multiple of 64 for efficiency
    val nLayer: Int = 12,
    val nHead: Int = 12,
    val nEmbed: Int = 768,
    val dropout: Float = 0.0f,
    val bias: Boolean = true // True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
)
