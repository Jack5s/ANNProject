import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

public  class Printer {
	public static void printDataset(DataSet datasetPrint) {
		int i = 0;
		for (DataSetRow dataSetRow : datasetPrint) {
			i++;
			double[] dataInputArray = dataSetRow.getInput();
			double demValue = dataInputArray[0];
			double ndviValue = dataInputArray[1];
			double[] dataOutputArray = dataSetRow.getDesiredOutput();
			double rainfallValue = dataOutputArray[0];
			System.out.println(i + ": " + demValue + ";" + ndviValue + ";" + rainfallValue);
		}
	}

	public static void printWeights(Double[] weightArray) {		
		for (int i = 0; i < weightArray.length; i++) {
			System.out.print(weightArray[i] + ";");
		}
		System.out.println("");
	}
}
