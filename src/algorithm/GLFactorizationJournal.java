package algorithm;

import datamodel.Triple;

/*
 * @(#)GLMatrixFactorizationJournal.java
 * 
 * Matrix factorization with generalized probability, the journal paper version.
 * Project: Matrix factorization for recommender systems.
 * Author: Fan Min and Zhuo-Lin Fu
 * www.fansmale.com
 * Email: minfan@swpu.edu.cn, minfanphd@163.com.
 * Created: December 10, 2019.
 * Last modified: Jan 16, 2019.
 */

//class 子类 extends 父类 {}
public class GLFactorizationJournal extends MatrixFactorization {

	// Triple[] glData;

	/**
	 * Add the variation to the boundaries during data transferring to avoid
	 * NaN.
	 */
	public static final double BOUNDARY_VARIATION = 0.0001;

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
	public GLFactorizationJournal(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, double paraTrainingFraction) {
		super(paraFilename, paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound, paraRatingUpperBound,
				paraTrainingFraction);
	}// Of the first constructor

	/**
	 ************************ 
	 * Transfer a rating value.
	 * 
	 * @param paraValue
	 *            The given value.
	 ************************ 
	 */
	public static double glTransform(double paraValue, double paraV, double paraLowerBound, double paraUpperBound) {
		if (paraValue < paraLowerBound + BOUNDARY_VARIATION) {
			paraValue = paraLowerBound + BOUNDARY_VARIATION;
		} // Of if

		if (paraValue > paraUpperBound - BOUNDARY_VARIATION) {
			paraValue = paraUpperBound - BOUNDARY_VARIATION;
		} // Of if

		double resultValue = -Math
				.log(Math.pow((paraUpperBound - paraLowerBound) / (paraValue - paraLowerBound), paraV) - 1);

		return resultValue;
	}// of glTransform

	/**
	 ************************ 
	 * Transfer the rating value back.
	 * 
	 * @param paraValue
	 *            The given value.
	 ************************ 
	 */
	public static double glInverseTransfer(double paraValue, double paraV, double paraLowerBound,
			double paraUpperBound) {
		double resultValue = (paraUpperBound - paraLowerBound) / Math.pow(Math.exp(-paraValue) + 1, 1 / paraV)
				+ paraLowerBound;

		return resultValue;
	}// of glReverseTransfer

	/**
	 ************************ 
	 * Transfer the whole data to GL data.
	 ************************ 
	 */
	public Triple[] glTransform() {
		Triple[] tempNewData = new Triple[trainingData.length];
		for (int i = 0; i < trainingData.length; i++) {
			tempNewData[i] = new Triple(trainingData[i].user, trainingData[i].item,
					glTransform(trainingData[i].rating, 1, ratingLowerBound, ratingUpperBound));
		} // Of for i

		return tempNewData;
	}// of glTransform

	/**
	 ************************ 
	 * Compute the MAE.
	 * 
	 * @return MAE of the current factorization.
	 ************************ 
	 */
	public double glMae() {
		double resultMae = 0;
		int tempTestCount = 0;

		for (int i = 0; i < testingData.length; i++) {
			int tempUserIndex = testingData[i].user;
			int tempItemIndex = testingData[i].item;
			double tempRate = testingData[i].rating - meanRating;

			double tempPrediction = predict(tempUserIndex, tempItemIndex);

			tempPrediction = glInverseTransfer(tempPrediction, 1.0, ratingLowerBound, ratingUpperBound);

			if (tempPrediction < ratingLowerBound) {
				tempPrediction = ratingLowerBound;
			} // Of if
			if (tempPrediction > ratingUpperBound) {
				tempPrediction = ratingUpperBound;
			} // of if

			double tempError = tempRate - tempPrediction;

			resultMae += Math.abs(tempError);
			// System.out.println("resultMae: " + resultMae);
			tempTestCount++;
		} // Of for i

		return (resultMae / tempTestCount);
	}// Of glMae

	/**
	 ************************ 
	 * Compute the RSME.
	 * 
	 * @return RSME of the current factorization.
	 ************************ 
	 */
	public double glRmse() {
		double resultRsme = 0;
		int tempTestCount = 0;

		for (int i = 0; i < testingData.length; i++) {
			int tempUserIndex = testingData[i].user;
			int tempItemIndex = testingData[i].item;
			double tempRate = testingData[i].rating - meanRating;

			double tempPrediction = predict(tempUserIndex, tempItemIndex);
			tempPrediction = glInverseTransfer(tempPrediction, 1.0, ratingLowerBound, ratingUpperBound);

			if (tempPrediction < ratingLowerBound) {
				tempPrediction = ratingLowerBound;
			} else if (tempPrediction > ratingUpperBound) {
				tempPrediction = ratingUpperBound;
			} // of if

			double tempError = tempRate - tempPrediction;
			resultRsme += tempError * tempError;
			tempTestCount++;
		} // Of for i

		return Math.sqrt(resultRsme / tempTestCount);
	}// Of glRmse

	/**
	 ************************ 
	 * Test the class.
	 ************************ 
	 */
	public static void test(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraLowerBound, double paraUpperBound, int paraRounds) {
		try {
			// Step 1. read the training and testing data
			GLFactorizationJournal tempGLF = new GLFactorizationJournal(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraLowerBound, paraUpperBound, 0.8);

			tempGLF.setParameters(5, 0.00004, 0.008, PQ_REGULAR);

			// Step 2. Initialize the feature matrices U and V
			tempGLF.initializeSubspaces(0.5);
			double tempMAE = tempGLF.glMae();
			double tempRSME = tempGLF.rsme();
			System.out.println("GLFactorization. Before training, MAE = " + tempMAE + ", RSME = " + tempRSME);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");
			tempGLF.trainingData = tempGLF.glTransform();

			tempGLF.train(paraRounds);

			// Step 4. Compute the accuracy
			tempMAE = tempGLF.glMae();
			tempRSME = tempGLF.rsme();
			System.out.println("MAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of test

	/**
	 ************************ 
	 * Test the class.
	 ************************ 
	 */
	public static void glTransferTest() {
		for (double tempValue = -10; tempValue <= 10; tempValue++) {
			double tempTransferedValue = glTransform(tempValue, 1.5, -10, 10);
			System.out.println("Value " + tempValue + ", transfered to " + tempTransferedValue);
		} // Of for
	}// Of glTransferTest

	/**
	 ************************ 
	 * Test the class.
	 ************************ 
	 */
	public static void glInverseTransferTest() {
		for (double tempValue = -5; tempValue <= 5; tempValue++) {
			double tempTransferedValue = glInverseTransfer(tempValue, 1.5, -10, 10);
			System.out.println("Value " + tempValue + ", transfered to " + tempTransferedValue);
		} // Of for
	}// Of glInverseTransferTest

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		test("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, 2000);
		// glTransferTest();
		// System.out.println("Inverse");
		// glInverseTransferTest();
		// glTransferOldTest();
	}// of main
}// Of class GLFactorization
