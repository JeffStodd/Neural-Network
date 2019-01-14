
public class Driver {

	public static void main(String[] args) {
		NN test = new NN( new int[]{3,2,1});
		test.loadWeights("D:/Eclipse Workspace/Neural Network/weights.txt");
		test.visualize();
		test.saveWeights("D:/Eclipse Workspace/Neural Network/weights.txt");
		//NN test2 = new NN(new int[] {3,2,1});
		//test2.loadWeights("C:/Users/jeffr/Desktop/weights.txt");
		//test2.visualize();
		
		test.forwardPropogation(new double[] {0.5,0.75,0.36});
	}

}
