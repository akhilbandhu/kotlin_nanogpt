package model

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
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import model.config.GPTConfig
import org.deeplearning4j.nn.conf.inputs.InputType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import kotlin.math.sqrt

class MultilayerPerceptron(private val config: GPTConfig) : SameDiffLayer() {

    override fun defineParameters(params: SDLayerParams) {
        // Define parameters for the first fully connected layer
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, config.nEmbed.toLong(), 4 * config.nEmbed.toLong())
        if (config.bias) {
            params.addBiasParam(DefaultParamInitializer.BIAS_KEY, 4 * config.nEmbed.toLong())
        }

        // Define parameters for the second fully connected layer
        params.addWeightParam("c_proj_weight", 4 * config.nEmbed.toLong(), config.nEmbed.toLong())
        if (config.bias) {
            params.addBiasParam("c_proj_bias", config.nEmbed.toLong())
        }
    }

    override fun initializeParameters(params: Map<String, INDArray>) {
        // Initialize the weights for the first fully connected layer with Xavier initialization
        if (params.containsKey(DefaultParamInitializer.WEIGHT_KEY)) {
            val weights = params[DefaultParamInitializer.WEIGHT_KEY]
            val fanIn = config.nEmbed.toDouble() // Number of input units, in the case of first layer
            val fanOut = (4 * config.nEmbed).toDouble() // Number of output units, in the case of first layer
            val xavierInitRange = sqrt(6.0 / (fanIn + fanOut))
            Nd4j.rand(weights?.shape(), Nd4j.getDistributions().createUniform(-xavierInitRange, xavierInitRange))
        }

        // Initialize bias to zero if required
        if (config.bias && params.containsKey(DefaultParamInitializer.BIAS_KEY)) {
            params[DefaultParamInitializer.BIAS_KEY]?.assign(0)
        }

        // Initialize the weights for the second fully connected layer with Xavier initialization
        if (params.containsKey("c_proj_weight")) {
            val cProjWeights = params["c_proj_weight"]
            val fanIn = (4 * config.nEmbed).toDouble() // Number of input units, in the case of second layer
            val fanOut = config.nEmbed.toDouble() // Number of output units, in the case of second layer
            val xavierInitRange = sqrt(6.0 / (fanIn + fanOut))
            if (cProjWeights != null) {
                Nd4j.rand(cProjWeights.shape(), Nd4j.getDistributions().createUniform(-xavierInitRange, xavierInitRange))
            }
        }

        // Initialize c_proj bias to zero if required
        if (config.bias && params.containsKey("c_proj_bias")) {
            params["c_proj_bias"]?.assign(0)
        }
    }

    override fun defineLayer(
        sd: SameDiff,
        layerInput: SDVariable,
        paramTable: Map<String, SDVariable>,
        mask: SDVariable?
    ): SDVariable {
        // First fully connected layer
        var x = sd.nn.linear(layerInput, paramTable[DefaultParamInitializer.WEIGHT_KEY], paramTable[DefaultParamInitializer.BIAS_KEY])

        // GELU activation
        x = sd.nn.gelu(x)

        // Second fully connected layer
        x = sd.nn.linear(x, paramTable["c_proj_weight"], paramTable["c_proj_bias"])

        // Dropout
        if (config.dropout > 0) {
            x = sd.nn.dropout(x, config.dropout)
        }

        return x
    }

    override fun getOutputType(layerIndex: Int, inputType: InputType): InputType {
        // Assuming the output type is the same as the input type
        return inputType
    }
}
