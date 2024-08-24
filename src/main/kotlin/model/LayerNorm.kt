package model

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.springframework.stereotype.Component

/**
 * ChatGPT trying to explain this to a 5 yr old:
 * Think of layer normalization like a game where everyone gets a turn to speak equally.
 * In a computer's brain, when it's learning or deciding something, some parts may talk too loudly or too softly.
 * Layer normalization is like a rule in the game that helps every part speak at a nice, even level, so the computer can understand things better and make smarter choices.
 * It's like making sure everyone in a circle gets to share their story without anyone being too loud or too quiet.
 *
 * One of the steps in a Transformer block is to apply layer normalization to this matrix.
 * This is an operation that normalizes the values in each column of the matrix separately.
 *
 * @param nDim Number of dimensions to the layer
 * @param bias If you need to include bias or not
 */

@Component
class LayerNorm(private val tf: Ops, private val ndim: Int, private val useBias: Boolean) {
    private val weight: Operand<TFloat32> = tf.variable(tf.ones(tf.constant(intArrayOf(ndim)), TFloat32::class.java))
    private val bias: Operand<TFloat32>? = if (useBias) tf.variable(tf.zeros(tf.constant(intArrayOf(ndim)), TFloat32::class.java)) else null

    fun forward(input: Operand<TFloat32>): Operand<TFloat32> {
        val axes = tf.constant(intArrayOf(-1))
        val mean = tf.math.mean(input, axes)
        val variance = tf.math.mean(tf.math.square(tf.math.sub(input, mean)), axes)
        val normalized = tf.math.div(
            tf.math.sub(input, mean),
            tf.math.sqrt(tf.math.add(variance, tf.constant(1e-5f)))
        )
        return if (bias != null) {
            tf.math.add(tf.math.mul(weight, normalized), bias)
        } else {
            tf.math.mul(weight, normalized)
        }
    }
}