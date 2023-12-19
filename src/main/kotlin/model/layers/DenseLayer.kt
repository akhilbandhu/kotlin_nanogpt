package model.layers

import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.types.TFloat32

class DenseLayer(private val inputSize: Int, private val outputSize: Int, private val useBias: Boolean, private val graph: Graph) {
    private lateinit var weights: Variable<TFloat32>
    private var bias: Variable<TFloat32>? = null
    fun build(tf: Ops) {
        weights = tf.withName("weight").variable(
            tf.constant(TFloat32.scalarOf(inputSize.toFloat()))
        )
        bias = if (useBias) tf.withName("bias").variable(
            tf.constant(TFloat32.scalarOf(outputSize.toFloat()))
        ) else null

        // Initialize weights and bias
//        val weightInit = tf.assign(weights, tf.random.normal(weights.shape(), 0.0f, 1.0f)) // Example: random normal initialization
//        val biasInit = bias?.let { tf.assign(it, tf.zeros(it.shape())) } // Bias initialized to zero
//
//        val graph = Graph()
//        // Run the initializers
//        Session(graph).use { session ->
//            session.runner().run(weightInit)
//            biasInit?.let { session.runner().run(it) }
//        }
    }

    fun call(tf: Ops, input: Operand<TFloat32>): Operand<TFloat32> {
        val output: Operand<TFloat32> = tf.linalg.matMul(input, weights)
        return bias?.let { tf.math.add(output, it) } ?: output
    }
}
