package model

import model.config.GPTConfig
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.core.Init.add
import org.tensorflow.types.TFloat32

class Block(config: GPTConfig) {
    private val ln1: LayerNorm = LayerNorm(config.nEmbed, includeBias = config.bias)
    private val attn: CausalSelfAttention = CausalSelfAttention(config)
    private val ln2: LayerNorm = LayerNorm(config.nEmbed, includeBias = config.bias)
    private val mlp: MultilayerPerceptron = MultilayerPerceptron(config)

    fun forward(x: Operand<TFloat32>): Operand<TFloat32> {
        var xVar = x
        xVar = xVar.apply {
            add(attn.forward(ln1.forward(xVar)))
        }
        xVar = xVar.asTensor().apply {
            add(mlp.forward(ln2.forward(xVar)))
        }
        return xVar
    }
}