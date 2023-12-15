package model

import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.math.Mean
import org.tensorflow.types.TFloat32

/**
 * ChatGPT trying to explain this to a 5 yr old:
 * Think of layer normalization like a game where everyone gets a turn to speak equally.
 * In a computer's brain, when it's learning or deciding something, some parts may talk too loudly or too softly.
 * Layer normalization is like a rule in the game that helps every part speak at a nice, even level, so the computer can understand things better and make smarter choices.
 * It's like making sure everyone in a circle gets to share their story without anyone being too loud or too quiet.
 *
 * One of the steps in a Transformer block is to apply layer normalization to this matrix.
 * This is an operation that normalizes the values in each column of the matrix separately.
 */

class LayerNorm(private val ndim: Int, private val includeBias: Boolean) {
    private var graph: Graph = Graph()
    private var tf: Ops = Ops.create(graph)

    private val weight: Variable<TFloat32> = tf.withName("weight").variable(
        tf.constant(TFloat32.scalarOf(ndim.toFloat()))
    )
    private val bias: Variable<TFloat32>? = if (includeBias) tf.withName("bias").variable(
        tf.constant(TFloat32.scalarOf(ndim.toFloat()))
    ) else null

    fun forward(input: Operand<TFloat32>): Operand<TFloat32> {
        val mean = tf.math.mean(input, tf.constant(ndim), Mean.keepDims(true))
        val variance = tf.math.mean(
            tf.math.squaredDifference(input, mean),
            tf.constant(ndim),
            Mean.keepDims(true)
        )

        val normalized = tf.math.div(
            tf.math.sub(input, mean),
            tf.math.sqrt(tf.math.add(variance, tf.constant(1e-5f)))
        )

        return tf.math.add(tf.math.mul(normalized, weight), bias ?: tf.constant(0f))
    }
}
