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
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.op.dtypes.Cast
import model.config.GPTConfig
import org.springframework.stereotype.Component

@Component
class GPT(private val tf: Ops, private val config: GPTConfig) {

    private val wte: Operand<TFloat32>
    private val wpe: Operand<TFloat32>
    private val blocks: List<Block>
    private val lnF: LayerNorm
    private val lmHead: Linear

    init {
        wte = tf.variable(tf.random.truncatedNormal(tf.constant(intArrayOf(config.vocabSize, config.nEmbed)), TFloat32::class.java))
        wpe = tf.variable(tf.random.truncatedNormal(tf.constant(intArrayOf(config.blockSize, config.nEmbed)), TFloat32::class.java))
        blocks = List(config.nLayer) { Block(tf, config) }
        lnF = LayerNorm(tf, config.nEmbed, config.bias)
        lmHead = Linear(tf, config.nEmbed, config.vocabSize, false)
    }

    fun forward(idx: Operand<TInt32>, targets: Operand<TInt32>? = null): Pair<Operand<TFloat32>, Operand<TFloat32>?> {
        val (b, t) = idx.shape().asArray()
        require(t <= config.blockSize) { "Cannot forward sequence of length $t, block size is only ${config.blockSize}" }

        val pos = tf.range(tf.constant(0), tf.constant(t.toInt()), tf.constant(1))
        var x: Operand<TFloat32> = tf.math.add(tf.gather(wte, idx, tf.constant(0)), tf.gather(wpe, pos, tf.constant(0))).asOutput()

        for (block in blocks) {
            x = block.forward(x)
        }
        x = lnF.forward(x)

        val logits = lmHead.forward(x)

        val loss = if (targets != null) {
            tf.nn.softmaxCrossEntropyWithLogits(tf.dtypes.cast(targets, TFloat32::class.java), logits).loss()
        } else {
            null
        }

        return Pair(logits, loss)
    }

    fun generate(idx: Operand<TInt32>, maxNewTokens: Int, temperature: Float = 1.0f, topK: Int? = null): Operand<TInt32> {
        var sequence = idx
        repeat(maxNewTokens) {
            val (logits, _) = forward(sequence)
            val lastLogits = tf.slice(logits, tf.constant(intArrayOf(-1, -1, 0)), tf.constant(intArrayOf(-1, 1, -1)))
            val scaledLogits = tf.math.div(lastLogits, tf.constant(temperature))
            
            val nextToken = if (topK != null) {
                val topkResult = tf.nn.topK(scaledLogits, tf.constant(topK))
                val sampledIndex = tf.random.statelessMultinomial(
                    tf.nn.softmax(topkResult.values()),
                    tf.constant(1),
                    tf.constant(intArrayOf(42, 42))  // Seed for reproducibility
                )
                tf.gather(topkResult.indices(), sampledIndex, tf.constant(0))
            } else {
                tf.random.statelessMultinomial(
                    tf.nn.softmax(scaledLogits),
                    tf.constant(1),
                    tf.constant(intArrayOf(42, 42))  // Seed for reproducibility
                )
            }
            
            sequence = tf.concat(listOf(sequence, tf.dtypes.cast(nextToken, TInt32::class.java)), tf.constant(1))
        }
        return sequence
    }
}