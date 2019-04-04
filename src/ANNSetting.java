import java.util.List;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.transfer.Sigmoid;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.util.TransferFunctionType;

public class ANNSetting {
	public static void setLayer(NeuralNetwork ann, int layerIndex, TransferFunction transferFunction) {
		Layer layer = ann.getLayerAt(layerIndex);
		List<Neuron> neuronList = layer.getNeurons();
		for (Neuron neuron : neuronList) {
			neuron.setTransferFunction(transferFunction);
		}
	}
}
