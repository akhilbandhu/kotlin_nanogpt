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
import org.tensorflow.op.linalg.BandPart
import org.tensorflow.op.linalg.Transpose
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.springframework.stereotype.Component
import model.config.GPTConfig

@Component
class CausalSelfAttention(private val tf: Ops, private val config: GPTConfig) {
    private val cAttn: Linear
    private val cProj: Linear

    init {
        require(config.nEmbed % config.nHead == 0)
        cAttn = Linear(tf, config.nEmbed, 3 * config.nEmbed, config.bias)
        cProj = Linear(tf, config.nEmbed, config.nEmbed, config.bias)
    }

    fun forward(x: Operand<TFloat32>): Operand<TFloat32> {
        val (b, t, c) = x.shape().asArray()
        
        val qkv = cAttn.forward(x)
        val q = tf.slice(qkv, tf.constant(intArrayOf(0, 0, 0)), tf.constant(intArrayOf(-1, -1, config.nEmbed)))
        val k = tf.slice(qkv, tf.constant(intArrayOf(0, 0, config.nEmbed)), tf.constant(intArrayOf(-1, -1, config.nEmbed)))
        val v = tf.slice(qkv, tf.constant(intArrayOf(0, 0, 2 * config.nEmbed)), tf.constant(intArrayOf(-1, -1, config.nEmbed)))

        val att = tf.linalg.matMul(q, tf.linalg.transpose(k, tf.constant(intArrayOf(0, 2, 1))))
        val scaledAtt = tf.math.div(att, tf.constant(Math.sqrt(config.nEmbed.toDouble() / config.nHead).toFloat()))
        val maskedAtt = tf.math.mul(
            scaledAtt,
            tf.linalg.bandPart(
                tf.ones(tf.constant(intArrayOf(t.toInt(), t.toInt())), TFloat32::class.java),
                tf.constant(-1L),
                tf.constant(0L)
            )
        )
        val attProbs = tf.nn.softmax(maskedAtt)

        val y = tf.linalg.matMul(attProbs, v)
        return cProj.forward(y)
    }
}
