package model

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.op.math.Add
import org.tensorflow.op.nn.Softmax
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.types.TFloat32
import org.springframework.stereotype.Component
import model.config.GPTConfig

@Component
class Block(private val tf: Ops, private val config: GPTConfig) {
    private val ln1 = LayerNorm(tf, config.nEmbed, config.bias)
    private val attn = CausalSelfAttention(tf, config)
    private val ln2 = LayerNorm(tf, config.nEmbed, config.bias)
    private val mlp = MLP(tf, config)

    fun forward(x: Operand<TFloat32>): Operand<TFloat32> {
        var output = tf.math.add(x, attn.forward(ln1.forward(x)))
        output = tf.math.add(output, mlp.forward(ln2.forward(output)))
        return output
    }
}
