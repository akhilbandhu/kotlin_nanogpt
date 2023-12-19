package model


import model.layers.DenseLayer
import model.layers.DropoutLayer
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.Activation
import org.tensorflow.types.TFloat32

/**
 * MLP, also known as matrix multiplication in machine learning
 *
 * After prodding ChatGPT:
 * Yes, at its core, an MLP (Multilayer Perceptron) largely involves matrix multiplication.
 * Each neuron's output is computed as a weighted sum of its inputs, which is a form of matrix multiplication.
 * These results are then typically passed through a non-linear function (like a sigmoid or ReLU function).
 * So, while matrix multiplication is a key part of how an MLP processes data,
 * it also involves additional steps like applying non-linear transformations and aggregating these computations across multiple layers.
 */

class MultilayerPerceptron(nEmbDim: Int, useBias: Boolean, dropoutRate: Float, graph: Graph) {
    private val tf: Ops = Ops.create(graph)
    private val cFcLayer: DenseLayer
    private val cProjLayer: DenseLayer
    private val dropoutLayer: DropoutLayer

    init {
        cFcLayer = DenseLayer(nEmbDim, 4 * nEmbDim, useBias, graph)
        cProjLayer = DenseLayer(4 * nEmbDim, nEmbDim, useBias, graph)
        dropoutLayer = DropoutLayer(dropoutRate, graph)
    }

    fun call(input: Operand<TFloat32>, training: Boolean): Operand<TFloat32> {
        val fcOutput = cFcLayer.call(input)
        val geluOutput = tf.nn.relu(fcOutput)
        val projOutput = cProjLayer.call(geluOutput)
        return dropoutLayer.call(projOutput, training)
    }
}
