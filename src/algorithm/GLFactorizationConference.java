package algorithm;

import datamodel.Triple;

/*
 * @(#)GLMatrixFactorizationJournal.java
 * 
 * Matrix factorization with generalized probability, the conference paper version.
 * Project: Matrix factorization for recommender systems.
 * Author: Fan Min and Zhuo-Lin Fu
 * www.fansmale.com
 * Email: minfan@swpu.edu.cn, minfanphd@163.com.
 * Created: December 10, 2019.
 * Last modified: Jan 16, 2020.
 */

public class GLFactorizationConference extends MatrixFactorization {

	/**
	 * Add the variation to the boundaries during data transferring to avoid
	 * NaN.
	 */
	public static final double BOUNDARY_VARIATION = 0.0001;
	
   /**
    * The mean rating after GL transformation.
    */
	double glMeanRating;
	
   /**
    * A constant to avoid too small values after transformation.
    */
	public static final double MULTIPLEX_TRANFERED_VALUE = 500;

	/**
	 ************************ 
	 * The first constructor.
	 * 
	 * @param paraFilename
	 *            The data filename.
	 * @param paraNumUsers
	 *            The number of users.
	 * @param paraNumItems
	 *            The number of items.
	 * @param paraNumRatings
	 *            The number of ratings.
	 ************************ 
	 */
	public GLFactorizationConference(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings, double paraRatingLowerBound,
			double paraRatingUpperBound, double paraTrainingFraction) {
		super(paraFilename, paraNumUsers, paraNumItems, paraNumRatings,
				paraRatingLowerBound, paraRatingUpperBound, paraTrainingFraction);
	}// Of the first constructor

	/**
	 ************************ 
	 * Transfer a rating value.
	 * 
	 * @param paraValue
	 *            The given value.
	 ************************ 
	 */
	public double glTransfer(double paraValue, double paraB, double paraV,
			double paraConstant1, double paraConstant2) {
		if (paraValue > 10 - BOUNDARY_VARIATION) {
			paraValue = 10 - BOUNDARY_VARIATION;
		} else if (paraValue < -10 + BOUNDARY_VARIATION) {
			paraValue = -10 + BOUNDARY_VARIATION;
		}// of if

		double resultValue = -1
				/ paraB
				* (Math.log(Math.pow(paraConstant1
						/ (paraValue + paraConstant2), paraV) - 1));

		return resultValue;
	}// of glTransfer

	/**
	 ************************ 
	 * Transfer the given data to GL data.
	 * @param paraData The given data. It should be the training data.
	 ************************ 
	 */
	public Triple[] glTransfer(Triple[] paraData) {
		int tempNumRatings = paraData.length;
		glMeanRating = 0;
		Triple[] tempNewData = new Triple[tempNumRatings];
		for (int i = 0; i < tempNumRatings; i++) {
			tempNewData[i] = new Triple(paraData[i].user, paraData[i].item, glTransfer(paraData[i].rating, 1000, 0.01, 20, 10));
			
			glMeanRating += tempNewData[i].rating;
			// System.out.println(tempNewData[i].rating);
		}// Of for i
		glMeanRating /= tempNumRatings;

		System.out.println("glMeanRating = " + glMeanRating);
		
		for (int i = 0; i < tempNumRatings; i++) {
			tempNewData[i].rating = (tempNewData[i].rating - glMeanRating) * MULTIPLEX_TRANFERED_VALUE;
		}// Of for i

		// Return
		return tempNewData;
	}// of glTransfer

	/**
	 ************************ 
	 * Compute the MAE.
	 * 
	 * @return MAE of the current factorization.
	 ************************ 
	 */
	public double glMae() {
		double resultMae = 0;

		for (int i = 0; i < testingData.length; i++) {
			int tempUserIndex = testingData[i].user;
			int tempItemIndex = testingData[i].item;
			//double tempRate = testingData[i].rating + glMeanRating;

			double tempPrediction = predict(tempUserIndex, tempItemIndex);

			//System.out.print("tempPrediction  = " + tempPrediction + " -> ");
			tempPrediction = glInverseTransfer(tempPrediction / MULTIPLEX_TRANFERED_VALUE + glMeanRating, ratingLowerBound, ratingUpperBound,
					1.0, 1.0, 1000, 0.01);
			//System.out.println("" + tempPrediction + "glMeanRating = " + glMeanRating);

			tempPrediction -= glMeanRating;
			if (tempPrediction < ratingLowerBound) {
				tempPrediction = ratingLowerBound;
			}  else if (tempPrediction > ratingUpperBound) {
				tempPrediction = ratingUpperBound;
			} // Of if

			double tempError = testingData[i].rating - tempPrediction;

			resultMae += Math.abs(tempError);
		} // Of for i

		return (resultMae / testingData.length);
	}// Of glMae

	/**
	 ************************ 
	 * @param paraX
	 * @return
	 ************************ 
	 */
	public static double glInverseTransfer(double paraValue) {
		double tempX = 0;
		double A = -10;
		double K = 10;
		double C = 1;
		double Q = 1;
		double B = 1000;
		double v = 0.01;

		tempX = A + (K - A) / (Math.pow((C + Q * Math.exp(-B * paraValue)), 1 / v));

		return tempX;
	}// of glInverseTransfer

	/**
	 ************************ 
	 * @param paraX
	 * @return
	 ************************ 
	 */
	public static double glInverseTransfer(double paraValue, double paraLowerBound, double paraUpperBound,
			double paraC, double paraQ, double paraB, double paraV) {
		double resultValue = 0;

		resultValue = paraLowerBound + (paraUpperBound - paraLowerBound) / (Math.pow((paraC + paraQ * Math.exp(-paraB * paraValue)), 1 / paraV));

		return resultValue;
	}// of glInverseTransfer
	
	/**
	 ************************ 
	 * Test the class.
	 ************************ 
	 */
	public static void test(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings, double paraLowerBound,
			double paraUpperBound, int paraRounds) {
		try {
			// Step 1. read the training and testing data
			GLFactorizationConference tempGLF = new GLFactorizationConference(
					paraFilename, paraNumUsers, paraNumItems, paraNumRatings,
					paraLowerBound, paraUpperBound, 0.8);

			tempGLF.setParameters(5, 0.001, 0.005, NO_REGULAR);

			// Step 2. Initialize the feature matrices U and V
			tempGLF.initializeSubspaces(0.1);
			double tempMAE = tempGLF.glMae();
			double tempRSME = tempGLF.rsme();
			System.out.println("GLFactorizationHenry. Before training, MAE = "
					+ tempMAE + ", RSME = " + tempRSME);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");
			tempGLF.trainingData = tempGLF.glTransfer(tempGLF.trainingData);
			
			//Triple[] tempData = tempGLF.trainingData;

			tempGLF.train(paraRounds);

			// Store it back.
			//tempGLF.trainingData = tempData;

			// Step 4. Compute the accuracy
			tempMAE = tempGLF.glMae();
			System.out.println("GLMAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of test

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		test("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10,
				10, 2000);
		// test("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455,
		// -10,
		// 10, 200);
		// glTransferTest();
		// System.out.println("Inverse");
		// glInverseTransferTest();
		// glTransferOldTest();
	}// of main
}// Of class GLFactorizationHenry
