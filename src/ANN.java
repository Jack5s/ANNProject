import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.transfer.Sigmoid;
import org.neuroph.nnet.Adaline;
import org.neuroph.nnet.learning.LMS;

public class ANN {
	private class DataProperty {
		public double max = -999;
		public double min = 999;
	}

	DataProperty demDataProperty = null;
	DataProperty ndviDataProperty = null;
	DataProperty rainfallDataProperty = null;
	private String inputFilePath = "C:\\Users\\s1061395\\Course\\I3 Project\\Data\\Text\\data.txt";
	private String outputNormalizeFilePath = "C:\\Users\\s1061395\\Course\\I3 Project\\Data\\Text\\normalizeData.csv";

	public void train() throws Exception {
		DataSet dataset = getAllData();
		System.out.println(dataset.size());
		DataSet[] datasetArray = splitDataset(dataset);
		DataSet trainDataset = datasetArray[0];
		DataSet testDataset = datasetArray[1];

		System.out.println("Train Data Count: " + trainDataset.size());
		System.out.println("Test Data Count: " + testDataset.size());

		DataSet normalizeDataset = normalizeDataset(dataset);
		writeDataset(normalizeDataset, "C:\\Users\\s1061395\\Course\\I3 Project\\Data\\Text\\normalizeData.csv");

		NeuralNetwork<LMS> adalineANN = new Adaline(2);
		adalineANN.randomizeWeights(new Random(System.currentTimeMillis()));
		Printer.printWeights(adalineANN.getWeights());

		LMS lms = new LMS();
		lms.setLearningRate(0.001);
		lms.setMaxError(0.0001);
		lms.setMaxIterations(10000);
		adalineANN.setLearningRule(lms);

		ANNSetting.setLayer(adalineANN, 0, new Sigmoid());
		ANNSetting.setLayer(adalineANN, 1, new Sigmoid());

		System.out.println("Start Train ANN...");

		adalineANN.learn(normalizeDataset);
		System.out.println("Train finished!");
		Printer.printWeights(adalineANN.getWeights());

		testData(adalineANN, testDataset);
		
		OLSMultipleLinearRegression r = new OLSMultipleLinearRegression();
	}

	private DataSet getAllData() throws Exception {
		File file = new File(inputFilePath);
		BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
		String dataLine = bufferedReader.readLine();
		demDataProperty = new DataProperty();
		ndviDataProperty = new DataProperty();
		rainfallDataProperty = new DataProperty();
		DataSet dataset = new DataSet(2, 1);
		while (dataLine != null) {
			String[] dataArray = dataLine.split(";");
			double demValue = Double.valueOf(dataArray[0]);
			double ndviValue = Double.valueOf(dataArray[1]);
			double rainfallValue = Double.valueOf(dataArray[2]);
			dataset.addRow(new DataSetRow(new double[] { demValue, ndviValue }, new double[] { rainfallValue }));
			if (demValue > demDataProperty.max) {
				demDataProperty.max = demValue;
			}
			if (demValue < demDataProperty.min) {
				demDataProperty.min = demValue;
			}

			if (ndviValue > ndviDataProperty.max) {
				ndviDataProperty.max = ndviValue;
			}
			if (ndviValue < ndviDataProperty.min) {
				ndviDataProperty.min = ndviValue;
			}

			if (rainfallValue > rainfallDataProperty.max) {
				rainfallDataProperty.max = rainfallValue;
			}
			if (rainfallValue < rainfallDataProperty.min) {
				rainfallDataProperty.min = rainfallValue;
			}
			// System.out.println(demValue + ";" + ndviValue + ";" + rainfallValue);
			dataLine = bufferedReader.readLine();
		}
		bufferedReader.close();
		return dataset;
	}

	private DataSet[] splitDataset(DataSet dataset) {
		int totalCount = dataset.size();
		boolean[] checkArray = new boolean[totalCount];
		int i;
		for (i = 0; i < totalCount; i++) {
			checkArray[i] = false;
		}
		int trainCount = (int) Math.round(checkArray.length * 0.7);

		Random random = new Random(System.currentTimeMillis());
		i = 0;
		while (i < trainCount) {
			int randomValue = random.nextInt(totalCount);
			boolean check = checkArray[randomValue];
			if (check == true) {
				randomValue = random.nextInt(totalCount);
			} else {
				checkArray[randomValue] = true;
				i++;
			}
		}

		DataSet trainDataset = new DataSet(2, 1);
		DataSet testDataset = new DataSet(2, 1);
		for (i = 0; i < totalCount; i++) {
			DataSetRow dataSetRow = dataset.get(i);
			double[] dataInputArray = dataSetRow.getInput();
			double demValue = dataInputArray[0];
			double ndviValue = dataInputArray[1];
			double[] dataOutputArray = dataSetRow.getDesiredOutput();
			double rainfallValue = dataOutputArray[0];
			boolean check = checkArray[i];
			if (check == true) {
				trainDataset
						.addRow(new DataSetRow(new double[] { demValue, ndviValue }, new double[] { rainfallValue }));
			} else {
				testDataset
						.addRow(new DataSetRow(new double[] { demValue, ndviValue }, new double[] { rainfallValue }));
			}
		}
		DataSet[] result = new DataSet[2];
		result[0] = trainDataset;
		result[1] = testDataset;
		return result;
	}

	private DataSet normalizeDataset(DataSet dataset) {
		DataSet normalizeDataset = new DataSet(2, 1);

		double demRange = demDataProperty.max - demDataProperty.min;
		double ndviRange = ndviDataProperty.max - ndviDataProperty.min;
		double rainfallRange = rainfallDataProperty.max - rainfallDataProperty.min;
		int i = 0;
		for (DataSetRow dataSetRow : dataset) {
			double[] dataInputArray = dataSetRow.getInput();
			double demValue = dataInputArray[0];
			double ndviValue = dataInputArray[1];
			double[] dataOutputArray = dataSetRow.getDesiredOutput();
			double rainfallValue = dataOutputArray[0];

			double demNormalizeValue = (demValue - demDataProperty.min) / demRange;
			double ndviNormalizeValue = (ndviValue - ndviDataProperty.min) / ndviRange;
			double rainfallNormalizeValue = (rainfallValue - rainfallDataProperty.min) / rainfallRange;
			normalizeDataset.addRow(new DataSetRow(new double[] { demNormalizeValue, ndviNormalizeValue },
					new double[] { rainfallNormalizeValue }));

			// System.out.println(demNormalizeValue + ";" + ndviNormalizeValue + ";" +
			// rainfallNormalizeValue);
		}

		return normalizeDataset;
	}

	private void testData(NeuralNetwork ann, DataSet testDataset) {
		double rainfallRange = rainfallDataProperty.max - rainfallDataProperty.min;
		double meanError = 0.0;
		for (int i = 0; i < testDataset.size(); i++) {
			DataSetRow resultDataRow = testDataset.get(i);
			ann.setInput(resultDataRow.getInput());
			ann.calculate();
			double[] networkOutput = ann.getOutput();
			double predictNormalizeValue = networkOutput[0];
			double predictValue = predictNormalizeValue * rainfallRange + rainfallDataProperty.min;
			DataSetRow dataRow = testDataset.get(i);
			double data = dataRow.getDesiredOutput()[0];

			// System.out.println("Input: " + data + " Output: " + predictValue);
			meanError += Math.abs(data - predictValue);
		}
		meanError /= testDataset.size();
		System.out.println("Mean Error: " + meanError);
	}

	private void writeDataset(DataSet datasetWrite, String filePath) throws Exception {
		File file = new File(filePath);
		BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));
		for (DataSetRow dataSetRow : datasetWrite) {
			double[] dataInputArray = dataSetRow.getInput();
			double demNormolizeValue = dataInputArray[0];
			double ndviNormolizeValue = dataInputArray[1];
			double[] dataOutputArray = dataSetRow.getDesiredOutput();
			double rainfallNormolizeValue = dataOutputArray[0];
			String dataLine = demNormolizeValue + ";" + ndviNormolizeValue + ";" + rainfallNormolizeValue + "\r\n";
			bufferedWriter.write(dataLine);
		}
		bufferedWriter.close();
	}

}
