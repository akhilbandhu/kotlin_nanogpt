package model.layers

import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

class DropoutLayer(private val rate: Float, private val graph: Graph) {
    private val tf: Ops = Ops.create(graph)

    fun call(input: Operand<TFloat32>, training: Boolean): Operand<TFloat32> {
        return if (training) {
            // Calculate the dropout mask
            val dropoutMask = tf.withName("dropoutMask").math.floor(
                tf.withName("dropoutAdd").math.add(
                    tf.withName("dropoutRandom").random.randomUniform(
                        tf.constant(input.asOutput().shape()),
                        TFloat32::class.javaObjectType
                    ),
                    tf.constant(1.0f - rate)
                )
            )
            tf.math.mul(input, dropoutMask)
        } else {
            input
        }
    }
}