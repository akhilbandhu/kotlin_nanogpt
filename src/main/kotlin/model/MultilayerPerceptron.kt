package model

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.op.math.Add
import org.tensorflow.op.math.Sub
import org.tensorflow.op.math.Mul
import org.tensorflow.op.math.Div
import org.tensorflow.op.nn.Softmax
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import model.config.GPTConfig
import org.springframework.stereotype.Component

@Component
class MLP(private val tf: Ops, private val config: GPTConfig) {
    private val cFc = Linear(tf, config.nEmbed, 4 * config.nEmbed, config.bias)
    private val cProj = Linear(tf, 4 * config.nEmbed, config.nEmbed, config.bias)

    fun forward(x: Operand<TFloat32>): Operand<TFloat32> {
        return cProj.forward(tf.math.tanh(cFc.forward(x)))  // Using tanh as an alternative to GELU
    }
}

class Linear(private val tf: Ops, inFeatures: Int, outFeatures: Int, useBias: Boolean) {
    private val weight: Operand<TFloat32> = tf.variable(tf.random.truncatedNormal(tf.constant(intArrayOf(inFeatures, outFeatures)), TFloat32::class.java))
    private val bias: Operand<TFloat32>? = if (useBias) tf.variable(tf.zeros(tf.constant(intArrayOf(outFeatures)), TFloat32::class.java)) else null

    fun forward(input: Operand<TFloat32>): Operand<TFloat32> {
        val output = tf.linalg.matMul(input, weight)
        return if (bias != null) tf.math.add(output, bias) else output
    }
}
