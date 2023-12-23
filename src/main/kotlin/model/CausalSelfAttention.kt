package model

/**
 * From https://bbycroft.net/llm
 * The self-attention layer is perhaps the heart of the Transformer and of GPT.
 * It's the phase where the columns in our input embedding matrix "talk" to each other.
 *
 * The self-attention layer is made up of several heads, and we'll focus on one of them for now.
 */
import model.config.GPTConfig
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms.softmax
import kotlin.math.sqrt

abstract class CausalSelfAttention(private val config: GPTConfig) : Layer() {
    // Example translation of key, query, value projections
    private val cAttn: DenseLayer = DenseLayer.Builder()
        .nIn(config.nEmbed)
        .nOut(3 * config.nEmbed)
        .activation(Activation.IDENTITY)
        .dropOut(config.dropout)
        .build()
    private val cProj: DenseLayer = DenseLayer.Builder()
        .nIn(config.nEmbed)
        .nOut(config.nEmbed)
        .activation(Activation.IDENTITY)
        .dropOut(config.dropout)
        .build()

    fun forward(input: INDArray): INDArray {
        // Get the shape of the input array
        val b = input.shape()[0] // Batch size
        val t = input.shape()[1] // Sequence length
        val c = input.shape()[2] // Embedding dimensionality

        // Apply the attention linear layer and split the result into query, key, and value
        val combined = cAttn(input) // Replace with your actual method or layer call
        val split = combined.reshape(b, t, 3, config.nHead, c / config.nHead)

        val q = split.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())
        val k = split.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all())
        val v = split.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.all())

        // Reshape and transpose the last two dimensions for q, k, and v
        val qReshaped = q.permute(0, 2, 1, 3) // (B, nHead, T, hs)
        val kReshaped = k.permute(0, 2, 1, 3) // (B, nHead, T, hs)
        val vReshaped = v.permute(0, 2, 1, 3) // (B, nHead, T, hs)
        // Apply causal attention mechanism
        // Note: This is a complex operation in the original GPT-2 and would need a more detailed implementation
        val attentionOutput = scaledDotProductAttention(qReshaped, kReshaped, vReshaped)

        // Output projection
        return cProj.activate(attentionOutput)
    }

    private fun scaledDotProductAttention(
        query: INDArray,
        key: INDArray,
        value: INDArray,
        attnMask: INDArray? = null,
        dropoutP: Double = 0.0,
        isCausal: Boolean = false,
        scale: Double? = null
    ): INDArray {
        val depth = query.shape().last().toDouble()
        val scaleFactor = scale ?: (1.0 / sqrt(depth))

        // Transpose the last two dimensions of key
        val keyT = key.permute(0, 2, 1)

        val attnWeight = query.mmul(keyT).mul(scaleFactor)

        // Apply causal mask
        if (isCausal) {
            val L = query.size(-2)
            val S = key.size(-2)
            val causalMask = Nd4j.triu(Nd4j.ones(L, S), 0).mul(-1e9)
            attnWeight.addi(causalMask)
        }
        // Apply attention mask if provided
        attnMask?.let {
            attnWeight.addi(it)
        }

        val softmaxWeights = softmax(attnWeight)
        return softmaxWeights.mmul(value)
    }
}
