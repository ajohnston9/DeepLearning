/**
 * Denoising Autoencoder
 * @author @yusugomori on Github
 *
 */

import java.util.Random;

public class dA {
	public int N;
	
	/**
	 * Number of visible units
	 */ 
	public int n_visible;
	
	/**
	 * Number of hidden units
	 */ 
	public int n_hidden;
	
	/**
	 * Set of weights for the denoising autoencoder
	 */
	public double[][] W;
	
	/**
	 * Set of biases for the hidden units
	 */
	public double[] hbias;
	
	/**
	 * Set of biases for the visible units. Corresponds to b'
	 */
	public double[] vbias;
	
	/**
	 * Random number generator used by the dA
	 */
	public Random rng;
	
	/**
	 * Get a uniform random number
	 * @param min the minimum value of the number
	 * @param max the maximum value of the number
	 */ 
	public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}
	
	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;
		
		int c = 0;
		double r;
		
		for(int i=0; i<n; i++) {
			r = rng.nextDouble();
			if (r < p) c++;
		}
		
		return c;
	}
	
	/**
	 * Computes tanh(x) for a given x
	 * @param x the input to tanh()
	 * @return the value of tanh() at x
	 */ 
	public static double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -x));
	}

	/**
	 * TODO: DESCRIPTION HERE
	 * @param N
	 * @param n_visible
	 * @param n_hidden
	 * @param W
	 * @param hbias
	 * @param vbias
	 * @param rng 
	 */
	public dA(int N, int n_visible, int n_hidden, 
			@Nullable double[][] W, double[] hbias, double[] vbias, @Nullable Random rng) {
		this.N = N;
		this.n_visible = n_visible;
		this.n_hidden = n_hidden;

		//Create a random generator with static seed if one is not provided
		if(rng == null)	{
			this.rng = new Random(1234);
		} else {
			this.rng = rng;
		}
		
		//If W is null, create a two-dimensional array 
		//of random numbers between -(1/n_visible) and (1/n_visible)		
		if(W == null) {
			this.W = new double[this.n_hidden][this.n_visible];
			double a = 1.0 / this.n_visible;
			
			for(int i=0; i<this.n_hidden; i++) {
				for(int j=0; j<this.n_visible; j++) {
					this.W[i][j] = uniform(-a, a); 
				}
			}	
		} else {
			this.W = W;
		}
		
		//If hbias is null create an array of zeros and assign it to hbias
		if(hbias == null) {
			this.hbias = new double[this.n_hidden];
			for(int i=0; i<this.n_hidden; i++) {
				this.hbias[i] = 0;
			}
		} else {
			this.hbias = hbias;
		}
		
		//If vbias is null create an array of zeros and assign it to vbias
		if(vbias == null) {
			this.vbias = new double[this.n_visible];
			for(int i=0; i<this.n_visible; i++) {
			   	this.vbias[i] = 0;
			}
		} else {
			this.vbias = vbias;
		}	
	}
	
	//Corrupt an input x
	public void get_corrupted_input(int[] x, int[] tilde_x, double p) {
		for(int i=0; i<n_visible; i++) {
			if(x[i] == 0) {
				tilde_x[i] = 0;
			} else {
				tilde_x[i] = binomial(1, p);
			}
		}
	}
	
	// Encode
	public void get_hidden_values(int[] x, double[] y) {
		for(int i=0; i<n_hidden; i++) {
			y[i] = 0;
			for(int j=0; j<n_visible; j++) {
				y[i] += W[i][j] * x[j];
			}
			y[i] += hbias[i];
			y[i] = sigmoid(y[i]);
		}
	}
	
	/**
	 * "Decodes" the signal. 
	 * @param y the input signal to reconstruct
	 * @param z the outputted reconstructed signal
	 */
	public void get_reconstructed_input(double[] y, double[] z) {
		for(int i=0; i<n_visible; i++) {
			z[i] = 0;
			for(int j=0; j<n_hidden; j++) {
				z[i] += W[j][i] * y[j];
			}
			z[i] += vbias[i];
			z[i] = sigmoid(z[i]);
		}
	}
	
	public void train(int[] x, double lr, double corruption_level) {
		int[] tilde_x = new int[n_visible];
		double[] y = new double[n_hidden];
		double[] z = new double[n_visible];
		
		double[] L_vbias = new double[n_visible];
		double[] L_hbias = new double[n_hidden];
		
		double p = 1 - corruption_level;
		
		get_corrupted_input(x, tilde_x, p);
		get_hidden_values(tilde_x, y);
		get_reconstructed_input(y, z);
		
		// vbias
		for(int i=0; i<n_visible; i++) {
			L_vbias[i] = x[i] - z[i];
			vbias[i] += lr * L_vbias[i] / N;
		}
		
		// hbias
		for(int i=0; i<n_hidden; i++) {
			L_hbias[i] = 0;
			for(int j=0; j<n_visible; j++) {
				L_hbias[i] += W[i][j] * L_vbias[j];
			}
			L_hbias[i] *= y[i] * (1 - y[i]);
			hbias[i] += lr * L_hbias[i] / N;
		}
		
		// W
		for(int i=0; i<n_hidden; i++) {
			for(int j=0; j<n_visible; j++) {
				W[i][j] += lr * (L_hbias[i] * tilde_x[j] + L_vbias[j] * y[i]) / N;
			}
		}
	}
	
	public void reconstruct(int[] x, double[] z) {
		double[] y = new double[n_hidden];
		
		get_hidden_values(x, y);
		get_reconstructed_input(y, z);
	}
	
	
	private static void test_dA() {
		Random rng = new Random(123);
		
		double learning_rate = 0.1;
		double corruption_level = 0.3;
		int training_epochs = 100;
		
		int train_N = 10;
		int test_N = 2;
		int n_visible = 20;
		int n_hidden = 5;
		
		int[][] train_X = {
			{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0}		
		};
		
		dA da = new dA(train_N, n_visible, n_hidden, null, null, null, rng);
		
		// train
		for(int epoch=0; epoch<training_epochs; epoch++) {
			for(int i=0; i<train_N; i++) {
				da.train(train_X[i], learning_rate, corruption_level);
			}
		}
		
		// test data
		int[][] test_X = {
			{1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0}				
		};
		
		double[][] reconstructed_X = new double[test_N][n_visible];
		
		// test
		for(int i=0; i<test_N; i++) {
			da.reconstruct(test_X[i], reconstructed_X[i]);
			for(int j=0; j<n_visible; j++) {
				System.out.printf("%.5f ", reconstructed_X[i][j]);
			}
			System.out.println();
		}
	}
	
	public static void main(String[] args) {
		test_dA();
	}
}
