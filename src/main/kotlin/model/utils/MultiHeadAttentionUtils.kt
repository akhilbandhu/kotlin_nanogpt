package model.utils

import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

class MultiHeadAttentionUtils {
    companion object {
        /**
         * Splits the last dimension of the tensor into (num_heads, depth).
         * Input shape: [batch_size, seq_length, num_features]
         * Output shape: [batch_size, num_heads, seq_length, depth]
         */
        fun splitHeads(tf: Ops, input: Operand<TFloat32>, numHeads: Int): Operand<TFloat32> {
            val shape = input.asOutput().shape()
            val batchSize = shape.size(0)
            val seqLength = shape.size(1)
            val numFeatures = shape.size(2)
            val depth = numFeatures / numHeads

            // Reshape to [batch_size, seq_length, num_heads, depth]
            val reshaped = tf.reshape(
                input,
                tf.constant(longArrayOf(batchSize, seqLength, numHeads.toLong(), depth))
            )

            // Transpose to [batch_size, num_heads, seq_length, depth]
            return tf.linalg.transpose(reshaped, tf.constant(intArrayOf(0, 2, 1, 3)))
        }

        /**
         * reshapeHeads function inverts the process done by splitHeads,
         * essentially collapsing the multi-head dimensions back into a single dimension.
         * This function is typically used after performing multi-head attention to prepare the output for subsequent layers in the model.
         */
        fun reshapeHeads(tf: Ops, input: Operand<TFloat32>, numHeads: Int): Operand<TFloat32> {
            // Assuming input shape is [batch_size, num_heads, seq_length, depth]
            val shape = input.asOutput().shape()
            val batchSize = shape.size(0)
            val seqLength = shape.size(2)
            val depth = shape.size(3)
            val numFeatures = numHeads * depth

            // Transpose back to [batch_size, seq_length, num_heads, depth]
            val transposed = tf.linalg.transpose(input, tf.constant(intArrayOf(0, 2, 1, 3)))

            // Reshape to [batch_size, seq_length, num_features]
            return tf.reshape(
                transposed,
                tf.constant(longArrayOf(batchSize, seqLength, numFeatures))
            )
        }

    }
}
