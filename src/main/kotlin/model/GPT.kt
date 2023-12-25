package model

import model.config.GPTConfig
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.deeplearning4j.nn.conf.layers.DropoutLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.MulOp
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

class GPT(config: GPTConfig) {
    private val model: ComputationGraphConfiguration
    private val computationGraph: ComputationGraph
    private val tokenEmbeddings: EmbeddingLayer
    private val positionEmbeddings: EmbeddingLayer
    private val dropout: DropoutLayer
    private val transformerBlocks: MutableList<Block>
    private val layerNorm: LayerNorm
    private val lmHead: DenseLayer

    companion object {
        // We dont need this for now
//        fun fromPretrained(modelName: String, config: GPTConfig): GPT {
//            val pretrainedModel =
//            val gpt = GPT(config)
//
//            // Assuming `pretrainedModel` is a ComputationGraph or similar object
//            val pretrainedParams = pretrainedModel.paramTable()
//
//            gpt.computationGraph.paramTable().forEach { (key, value) ->
//                if (pretrainedParams.containsKey(key)) {
//                    value.assign(pretrainedParams[key]) // Copy weights
//                }
//            }
//
//            return gpt
//        }
    }


    init {
        tokenEmbeddings = EmbeddingLayer.Builder()
            .nIn(config.vocabSize)
            .nOut(config.nEmbed)
            .build()

        positionEmbeddings = EmbeddingLayer.Builder()
            .nIn(config.blockSize)
            .nOut(config.nEmbed)
            .build()

        dropout = DropoutLayer.Builder(config.dropout).build()

        transformerBlocks = MutableList(config.nLayer) { Block(config) }
        layerNorm = LayerNorm(config.nEmbed, config.bias)

        lmHead = DenseLayer.Builder()
            .nIn(config.nEmbed)
            .nOut(config.vocabSize)
            .biasInit(0.0)
            .build()

        val confBuilder = NeuralNetConfiguration.Builder()
        // Here we need to build the ComputationGraphConfiguration
        val modelConfig = confBuilder.graphBuilder()
        model = modelConfig.build()
        computationGraph = ComputationGraph(model)
        computationGraph.init()

        initWeights()
    }

    // Methods for forward pass, parameter initialization, etc.

    fun forward(idx: INDArray, targets: INDArray? = null): Pair<INDArray, INDArray?> {
        val (b, t) = idx.shape()
//        require(t <= config.blockSize) { "Cannot forward sequence of length $t, block size is only ${config.blockSize}" }

        val pos = Nd4j.arange(t.toDouble())

        // Manual embedding lookup
        val tokEmbWeights = tokenEmbeddings.getParam("W")
        val tokEmb = Nd4j.pullRows(tokEmbWeights, 1, idx.toIntVector().size)

        val posEmbWeights = positionEmbeddings.getParam("W")
        val posEmb = Nd4j.pullRows(posEmbWeights, 1, pos.toIntVector().size)

        var x = tokEmb.add(posEmb)

        // Applying dropout manually
        val dropoutMask = Nd4j.rand(x.shape()).gt(config.dropout)
        Nd4j.getExecutioner().execAndReturn(MulOp(x, dropoutMask, x, x.length().toInt()))

        // Forward pass through each transformer block
        transformerBlocks.forEach { block ->
            x = block.forward(x)
        }
        x = layerNorm.forward(x)

        // Calculate logits and potentially loss
        val logits = lmHead.activate(x)
        val loss = if (targets != null) {
            Nd4j.getExecutioner().execAndReturn(LossFunctions.LossFunction.MCXENT, logits, targets)
        } else {
            null
        }

        return Pair(logits, loss)
    }


    private fun getNumParams(nonEmbedding: Boolean = true): Long {
        var numParams = computationGraph.paramTable().values.sumOf { it.length().toLong() }

        if (nonEmbedding) {
            // Subtract the number of parameters in the position embeddings layer
            val positionEmbeddingParams = computationGraph.getParam("wpe") // Assuming "wpe" is the name of the position embeddings layer
            numParams -= positionEmbeddingParams.length()
        }

        return numParams
    }

    private fun initWeights() {
        computationGraph.layers.forEach { layer ->
            val params = layer.paramTable()
            params?.forEach { (key, value) ->
                when {
                    key.equals("W", ignoreCase = true) && layer is DenseLayer -> {
                        // Initialize weights for DenseLayer (Linear in PyTorch)
                        value.assign(Nd4j.randn(*value.shape()).muli(0.02))
                    }
                    key.equals("b", ignoreCase = true) && layer is DenseLayer -> {
                        // Initialize biases for DenseLayer
                        value.assign(Nd4j.zeros(value.shape().size))
                    }
                    key.equals("W", ignoreCase = true) && layer is EmbeddingLayer -> {
                        // Initialize weights for EmbeddingLayer
                        value.assign(Nd4j.randn(*value.shape()).muli(0.02))
                    }
                }
            }
        }
    }

    private fun cropBlockSize(newBlockSize: Int) {
        // Adjust the position embeddings
        val posEmbWeights = positionEmbeddings.getParam("W")
        val croppedPosEmbWeights = posEmbWeights.get(NDArrayIndex.interval(0, newBlockSize.toLong()))
        positionEmbeddings.setParam("W", croppedPosEmbWeights)

        // Adjust attention masks in transformer blocks if necessary
        transformerBlocks.forEach { block ->
            block.adjustForBlockSize(newBlockSize)
        }
    }

    fun configureOptimizers(weightDecay: Double, learningRate: Double, betas: Pair<Double, Double>): Optimizer {
        // Assuming you're using an Adam optimizer
        val optimizer = Adam.Builder()
            .learningRate(learningRate)
            .beta1(betas.first)
            .beta2(betas.second)
            // Set weight decay if necessary
            .build()

        // Apply this optimizer to your computation graph
        computationGraph

        return optimizer
    }

    @JvmOverloads
    fun generate(startingSequence: INDArray, maxNewTokens: Int, temperature: Double = 1.0, topK: Int? = null): INDArray {
        var sequence = startingSequence.dup()

        for (i in 0 until maxNewTokens) {
            val (logits, _) = forward(sequence)

            // Apply temperature scaling
            logits.divi(temperature)

            // Apply Top-K filtering if necessary
            if (topK != null) {
                applyTopKFiltering(logits, topK)
            }

            // Convert logits to probabilities
            val probabilities = Nd4j.getExecutioner().exec(SoftMax(logits))

            // Sample from the distribution
            val sampledIndex = sampleFromDistribution(probabilities)

            // Append sampled index to sequence
            sequence = Nd4j.concat(1, sequence, sampledIndex)
        }

        return sequence
    }

    private fun sampleFromDistribution(probabilities: INDArray): INDArray {
        val sampledIndices = Nd4j.create(probabilities.rows(), 1)

        for (i in 0..<probabilities.rows()) {
            val probRow = probabilities.getRow(i)

            // Sample a single value based on the provided probabilities
            val cumulativeProb = probRow.cumsum(0)
            val randomValue = Nd4j.rand(1).getDouble(0) * cumulativeProb.getDouble(cumulativeProb.length() - 1)
            val sampledIndex = cumulativeProb.gt(randomValue).argMax()

            sampledIndices.putScalar(i.toLong(), sampledIndex.id.toDouble())
        }

        return sampledIndices
    }

    private fun applyTopKFiltering(logits: INDArray, k: Int) {
        // For each row in the logits (assuming batch dimension is the first)
        for (i in 0 until logits.rows()) {
            val logitRow = logits.getRow(i)

            // Get the top K values and their indices
            val sortedIndices = Nd4j.sortWithIndices(logitRow, 1, false).second // false for descending
            val topKIndices = sortedIndices.get(NDArrayIndex.interval(0, k.toLong()))

            // Create a mask that is True for top K values and False elsewhere
            val mask = Nd4j.createUninitialized(logitRow.shape().size)
            topKIndices.forEach { index -> mask.putScalar(index.toInt(), 1.0) }

            // Apply mask - set values not in the top K to a small number
            logitRow.muli(mask).addi(mask.rsub(1).muli(-1e10))
        }
    }

}