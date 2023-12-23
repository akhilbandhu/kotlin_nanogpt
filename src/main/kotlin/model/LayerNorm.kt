package model

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

import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.layers.normalization.BatchNormalization
import org.nd4j.linalg.activations.Activation
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.types.TFloat32

abstract class LayerNorm(ndim: Int, bias: Boolean) : Layer() {

    private var weight: Variable<TFloat32>
    private var bias: Variable<TFloat32>?

    init {
        // Assuming a TensorFlow-like syntax for variable creation
        val tf = Ops.create()
        weight =  tf.withName("weight").variable(
            tf.constant(TFloat32.scalarOf(ndim.toFloat()))
        )

        if (bias) {
            this.bias = tf.withName("bias").variable(tf.constant(TFloat32.scalarOf(ndim.toFloat())))
        } else {
            this.bias = null
        }
    }

    fun activate(input: Tensor, training: Boolean): Tensor {
        val tf = Ops.create()

        // Apply layer normalization
        val normalizedInput = BatchNormalization
            .activation(Activation.IDENTITY)
            .nIn(ndim)
            .nOut(ndim)
            .build()
            .activate(input)

        // Apply scaling and bias (if applicable)
        val scaledInput = tf.math.mul(normalizedInput, weight.read(tf))
        return if (bias != null) {
            tf.math.add(scaledInput, bias!!.read(tf))
        } else {
            scaledInput
        }
    }
}
