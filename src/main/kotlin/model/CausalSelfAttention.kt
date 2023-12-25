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
import org.deeplearning4j.nn.conf.layers.DropoutLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms.softmax
import kotlin.math.sqrt

class CausalSelfAttention(config: GPTConfig) {
    private val cAttn: DenseLayer
    private val cProj: DenseLayer
    private val attnDropout: DropoutLayer
    private val residDropout: DropoutLayer
    private val nHead: Int
    private val nEmb: Int
    private val dropout: Double
    private val bias: INDArray
    private val cAttnWeight: INDArray = Nd4j.rand(3 * config.nEmbed, config.nEmbed) // Example initialization
    private val cAttnBias: INDArray? = if (config.bias) Nd4j.rand(3 * config.nEmbed) else null
    private val cProjWeight: INDArray = Nd4j.rand(config.nEmbed, config.nEmbed) // Example initialization
    private val cProjBias: INDArray? = if (config.bias) Nd4j.rand(config.nEmbed) else null

    init {
        require(config.nEmbed % config.nHead == 0) { "Embedding dimension must be divisible by number of heads." }

        nHead = config.nHead
        nEmb = config.nEmbed
        dropout = config.dropout

        cAttn = DenseLayer.Builder()
            .nIn(nEmb)
            .nOut(3 * nEmb)
            .biasInit(0.0)
            .build()

        cProj = DenseLayer.Builder()
            .nIn(nEmb)
            .nOut(nEmb)
            .biasInit(0.0)
            .build()

        attnDropout = DropoutLayer.Builder(dropout).build()
        residDropout = DropoutLayer.Builder(dropout).build()

        // Causal mask creation
        bias = lowerTriangularMatrix(config.blockSize)
    }

    private fun dropout(x: INDArray, p: Double): INDArray {
        if (p <= 0.0) return x
        val mask = Nd4j.rand(x.shape().size).lt(p)
        return x.mul(mask).div(1.0 - p)
    }

    fun forward(input: INDArray): INDArray {
        val B = input.size(0)
        val T = input.size(1)
        val C = input.size(2)

        // Manual linear transformation for cAttn
        val combined = input.reshape(B * T, C).mmul(cAttnWeight.transpose())
        if (cAttnBias != null) combined.addiRowVector(cAttnBias)
        val reshapedCombined = combined.reshape(B, T, 3 * C)

        // Splitting and reshaping for multi-head attention
        val splitSize = C / nHead
        val q = reshapedCombined.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, splitSize))
        val k = reshapedCombined.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(splitSize, 2 * splitSize))
        val v = reshapedCombined.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(2 * splitSize, 3 * splitSize))

        // Reshape and transpose for multi-head attention
        val reshapeAndTranspose = { arr: INDArray ->
            val newShape = longArrayOf(B, T, nHead.toLong(), splitSize)
            arr.reshape(*newShape).permute(0, 2, 1, 3)
        }
        val qTransposed = reshapeAndTranspose(q)
        val kTransposed = reshapeAndTranspose(k)
        val vTransposed = reshapeAndTranspose(v)

        // Attention mechanism
        val y = scaledDotProductAttention(
            qTransposed,
            kTransposed,
            vTransposed,
            null,
            dropout,
            true
        )

        // Re-assemble and apply output projection
        val yReassembled = y.permute(0, 2, 1, 3).reshape(B, T, C)
        val projected = yReassembled.mmul(cProjWeight.transpose())
        if (cProjBias != null) projected.addiRowVector(cProjBias)

        // Apply dropout to the output
        return dropout(projected, dropout)
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

    private fun lowerTriangularMatrix(size: Int): INDArray {
        val matrix = Nd4j.zeros(size, size)
        for (i in 0..<size) {
            for (j in 0..i) {
                matrix.putScalar(intArrayOf(i, j), 1.0)
            }
        }
        return matrix
    }
}
