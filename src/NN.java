import java.awt.Graphics;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Stack;
import java.util.StringTokenizer;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class NN extends JPanel{

	private int[] layerTopology;

	private double [][] layers;
	private double [] inputs; //pointer
	private double [] outputs; //pointer

	private double [][][] weights;
	private double [] biases;
	private double [][] biasWeights;

	public NN(int[] topology)
	{
		layerTopology = topology;


		layers = new double[layerTopology.length][];
		for(int i = 0; i < layerTopology.length; i++)
		{
			layers[i] = new double[layerTopology[i]];
		}

		inputs = layers[0];
		outputs = layers[layerTopology.length-1];

		initWeights();
		initBiases();
	}

	public void initWeights()
	{
		weights = new double[layerTopology.length-1][][]; // length -2 or -1?
		//[numLayers][numNeurons][numWeights]
		for(int i = 0; i < weights.length; i++) //i < weights.length -1 exclude output having weights pointing to nothing
		{
			weights[i] = new double[layerTopology[i]][];
			for(int j = 0; j < weights[i].length; j++)
			{
				weights[i][j] = new double[layerTopology[i + 1]];
				for(int k = 0; k < weights[i][j].length; k++)
					weights[i][j][k] = Math.random();
			}	
		}
	}

	public void initBiases()
	{
		biases = new double[layerTopology.length-1];
		for(int i = 0; i < biases.length; i++)
			biases[i] = 1;
		biasWeights = new double[layerTopology.length-1][];
		for(int i = 0; i < biasWeights.length; i++)
		{
			biasWeights[i] = new double[layerTopology[i]];
			for(int j = 0; j < biasWeights[i].length; j++)
				biasWeights[i][j] = Math.random();
		}
	}

	public void saveWeights(String directory)
	{
		File save = new File("weights.txt");
		String str = "";

		for(int i = 0; i < weights.length; i++) //i < weights.length -1 exclude output having weights pointing to nothing
		{
			for(int j = 0; j < weights[i].length; j++)
			{
				for(int k = 0; k < weights[i][j].length; k++)
				{
					str += weights[i][j][k];
					if(k != weights[i][j].length-1)
						str += " ";
				}
				str += "\t";
			}
			str += "\n";
		}
		
		for(int i = 0; i < biasWeights.length; i++)
		{
			for(int j = 0; j < biasWeights[i].length; j++)
			{
				str += biasWeights[i][j];
					if(j != biasWeights[i].length - 1)
						str += " ";
			}
			str += "\n";
		}
			

		BufferedWriter writer = null;
		try
		{
			writer = new BufferedWriter( new FileWriter( save));
			writer.write( str);
			writer.close();
		}
		catch ( IOException e)
		{

		} 

	}

	public void loadWeights(String directory)
	{
		File load = new File(directory);
		BufferedReader reader = null;
		String line = "";
		Stack<String> stack = new Stack<String>();
		try
		{
			reader = new BufferedReader(new FileReader(load));
			while ((line = reader.readLine()) != null)
			{
				//System.out.println(line);
				StringTokenizer st = new StringTokenizer(line);
				while(st.hasMoreTokens())
					stack.add(st.nextToken());

				//System.out.println("next");
			}
			reader.close();
		}
		catch ( IOException e)
		{

		} 
		Collections.reverse(stack);
		for(int i = 0; i < weights.length; i++) //i < weights.length -1 exclude output having weights pointing to nothing
			for(int j = 0; j < weights[i].length; j++)
				for(int k = 0; k < weights[i][j].length; k++)
					weights[i][j][k] = Double.parseDouble(stack.pop());

		for(int i = 0; i < biasWeights.length; i++)
			for(int j = 0; j < biasWeights[i].length; j++)
				biasWeights[i][j] = Double.parseDouble(stack.pop());


	}

	public double[] forwardPropogation(double[] input)
	{
		inputs = input;
		layers[0] = input;
		for(int i = 1; i < layers.length; i++)
		{
			for(int j = 0; j < layers[i].length; j++)
			{
				System.out.print("Layer " + (i-1) + ": ");
				for(double var: layers[i-1])
					System.out.print(var + " ");
				System.out.println();

				double [] temp = new double[layers[i-1].length];
				for(int k = 0; k < layers[i-1].length; k++)
				{
					temp[k] = weights[i-1][k][j];
				}
				System.out.print("Weights [" + (i-1) + "][k][" + "[" + (j) + "]: ");
				for(double var: temp)
					System.out.print(var + " ");
				System.out.println();
				layers[i][j] = sigmoid(dot(layers[i-1], temp) + biases[i-1] * biasWeights[i-1][j]); //wrong arrays being dot? also check if += bias or += bias * weight
			}
		}
		System.out.print("Layer " + (layers.length-1) + ": ");
		for(double var: layers[layers.length-1])
			System.out.print(var + " ");
		System.out.println();
		outputs = layers[layers.length-1];
		return outputs;
	}

	public void backPropogation(double [] expected)
	{
		/*
		 * self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
		 */
		
		double error = 0;
		
		for(int i = 0; i < outputs.length; i++)
			error += 0.5 * (Math.pow(outputs[i] - expected[i],2));
		
		double errorDelta = 0;
	}

	public static double sigmoidPrime(double x)
	{
		return x * (1-x);
	}

	public static double sigmoid(double x)
	{
		return 1/(1 + Math.exp(-x));
	}

	public static double dot(double[] a, double[] b)
	{
		if(a.length != b.length)
		{
			System.out.println("SIZE MISMATCH ERROR");
			return -1;
		}
		else
		{
			double sum = 0;
			for (int i = 0; i < a.length; i++) {
				sum += a[i] * b[i];
			}
			return sum;
		}
	}

	public void paint(Graphics g)
	{
		for(int i = 0 ; i < layers.length; i++)
		{
			for(int j = 0; j < layers[i].length; j++)
			{
				int x = (int) (100 + (double)(i)/layers.length * 1080);
				int y = (int) (25 + (double)(j)/layers[i].length * 720);
				g.fillOval(x, y, 25, 25);
			}
		}

		for(int i = 0; i < weights.length; i++) //i < weights.length -1 exclude output having weights pointing to nothing
		{
			for(int j = 0; j < weights[i].length; j++)
			{
				for(int k = 0; k < weights[i][j].length; k++)
				{
					int x = (int) (100 + (double)(i)/layers.length * 1080);
					int y = (int) (25 + (double)(j)/layers[i].length * 720);

					int x2 = (int) (100 + (double)(i+1)/layers.length * 1080);
					int y2 = (int) (25 + (double)(k)/layers[i+1].length * 720);
					g.drawLine(x + 25/2,y + 25/2,x2 + 25/2,y2 + 25/2);
					g.drawString(""+ (float)weights[i][j][k], (x+x2)/2, (y+y2)/2);
				}
			}	
		}
	}

	public void visualize()
	{
		JFrame window = new JFrame();
		window.setBounds(150, 150, 1100, 750);
		window.setLayout(null);
		this.setSize(1080, 720);
		window.setEnabled(true);
		window.setVisible(true);
		window.add(this);
	}

	public int[] getLayerTopology() {
		return layerTopology;
	}

	public void setLayerTopology(int[] layerTopology) {
		this.layerTopology = layerTopology;
	}

	public double[][] getLayers() {
		return layers;
	}

	public void setLayers(double[][] layers) {
		this.layers = layers;
	}

	public double[] getInputs() {
		return inputs;
	}

	public void setInputs(double[] inputs) {
		this.inputs = inputs;
	}

	public double[] getOutputs() {
		return outputs;
	}

	public void setOutputs(double[] outputs) {
		this.outputs = outputs;
	}

	public double[][][] getWeights() {
		return weights;
	}

	public void setWeights(double[][][] weights) {
		this.weights = weights;
	}

	public double[] getBiases() {
		return biases;
	}

	public void setBiases(double[] biases) {
		this.biases = biases;
	}


}
