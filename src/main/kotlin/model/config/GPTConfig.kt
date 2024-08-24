package model.config

data class GPTConfig(
    val blockSize: Int = 1024,
    val vocabSize: Int = 50304,
    val nLayer: Int = 12,
    val nHead: Int = 12,
    val nEmbed: Int = 768,
    val dropout: Float = 0.0f,
    val bias: Boolean = true
)