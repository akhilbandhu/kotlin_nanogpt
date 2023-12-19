package model

import model.config.GPTConfig
import model.layers.DenseLayer
import model.layers.DropoutLayer
import model.utils.MultiHeadAttentionUtils
import model.utils.MultiHeadAttentionUtils.Companion.reshapeHeads
import model.utils.MultiHeadAttentionUtils.Companion.splitHeads
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.Softmax
import org.tensorflow.types.TFloat32
import kotlin.math.sqrt

/**
 * From https://bbycroft.net/llm
 * The self-attention layer is perhaps the heart of the Transformer and of GPT.
 * It's the phase where the columns in our input embedding matrix "talk" to each other.
 *
 * The self-attention layer is made up of several heads, and we'll focus on one of them for now.
 */
class CausalSelfAttention(private val config: GPTConfig) {
    private val graph: Graph = Graph()
    private val tf: Ops = Ops.create(graph)

    private val cAttn: DenseLayer // Custom dense layer for key, query, value projections
    private val cProj: DenseLayer // Custom dense layer for output projection
    private val attnDropout: DropoutLayer // Custom dropout layer
    private val residDropout: DropoutLayer // Custom dropout layer

    private val nHead: Int = config.nHead
    private val nEmbd: Int = config.nEmbd
    private val dropout: Float = config.dropout

    private val bias: Operand<TFloat32> // Custom implementation for causal mask

    init {
        // Initialization of layers and causal mask
        cAttn = DenseLayer(nEmbd, 3 * nEmbd, config.bias)
        cProj = DenseLayer(nEmbd, nEmbd, config.bias)
        attnDropout = DropoutLayer(dropout)
        residDropout = DropoutLayer(dropout)

        // Initialize bias here (causal mask)
    }

    fun forward(input: Operand<TFloat32>): Operand<TFloat32> {
        // Assuming input shape is (Batch, Time, Channels)
        val (B, T, C) = input.shape() // Extract batch size, sequence length, embedding dim

        // Split the projections into query, key, value
        val (q, k, v) = splitHeads((input), B, T) // Custom function to split and rearrange heads

        // Apply causal self-attention
        val y = if (config.useFlashAttention) {
            null
        } else {
            // Manual implementation of causal attention
            val att = tf.linalg.matMul(q, k, transposeB = true) * tf.constant(1.0 / sqrt(nEmbd.toDouble()))
            val maskedAtt = att.maskedFill(bias, Float.NEGATIVE_INFINITY) // Custom method for masking
            val softmaxAtt = Softmax.create(tf, maskedAtt, axis = -1)
            val droppedOutAtt = DropoutLayer(softmaxAtt)
            tf.linalg.matMul(droppedOutAtt, v)
        }

        // Reshape and project output
        val reshapedY = reshapeHeads(y, B, T) // Custom function to reshape heads back
        return DropoutLayer(DenseLayer(reshapedY))
    }
}