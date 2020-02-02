package algorithm;

import java.util.Arrays;

import common.*;
import datamodel.*;

/**
 * Incremental learning for matrix factorization with GL transformation <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: December 3, 2019.<br>
 *       Last modified: January 30, 2020.
 * @version 1.0
 */

public class MF2DBooleanIncrementalGLG extends MF2DBooleanIncremental {
	/**
	 * The original data.
	 */
	RatingSystem2DBoolean originalDataset;

	/**
	 * Add the variation to the boundaries during data transferring to avoid
	 * NaN.
	 */
	public static final double BOUNDARY_VARIATION = 0.000001;

	/**
	 * The lower bound of the rating value.
	 */
	protected double ratingLowerBound;

	/**
	 * The upper bound of the rating value.
	 */
	protected double ratingUpperBound;

	/**
	 * The journal version.
	 */
	public static final int GL_JOURNAL = 0;

	/**
	 * The conference version.
	 */
	public static final int GL_CONFERENCE = 1;

	/**
	 * The parameter for data transform.
	 */
	int algorithm;

	/**
	 * The parameter for data transform.
	 */
	double parameterV = 0.01;

	/**
	 * The parameter for data transform.
	 */
	double parameterB = 1000;

	/**
	 * The parameter for data transform.
	 */
	double constant1 = 20;

	/**
	 * The parameter for data transform.
	 */
	double constant2 = 10;

	/**
	 ************************ 
	 * The first constructor.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 * @param paraDataTransformAlgorithm
	 *            The data transform algorithm index.
	 ************************ 
	 */
	public MF2DBooleanIncrementalGLG(RatingSystem2DBoolean paraDataset,
			int paraDataTransformAlgorithm) {
		super(paraDataset);

		originalDataset = paraDataset;

		ratingLowerBound = originalDataset.getRatingLowerBound();
		ratingUpperBound = originalDataset.getRatingUpperBound();

		System.out.println("ratingBounds are: " + ratingLowerBound + ", " + ratingUpperBound);

		dataset = new RatingSystem2DBoolean(originalDataset);

		algorithm = paraDataTransformAlgorithm;
		parameterV = 1.5;
		// algorithm = GL_CONFERENCE;

		// Now convert ratings of the dataset.
		System.out.println("Data converting ...");
		double tempValue;
		for (int i = 0; i < dataset.data.length; i++) {
			for (int j = 0; j < dataset.data[i].length; j++) {
				tempValue = dataset.data[i][j].rating;
				if (algorithm == GL_JOURNAL) {
					dataset.data[i][j].rating = glTransformJournal(tempValue);
				} else {
					dataset.data[i][j].rating = glTransformConference(tempValue);
				} // Of if
			} // Of for j
		} // Of for i
	}// Of the first constructor

	/**
	 ************************ 
	 * Transform a rating value. The journal version.
	 * 
	 * @param paraValue
	 *            The given value.
	 ************************ 
	 */
	public double glTransformJournal(double paraValue) {
		if (paraValue < ratingLowerBound + BOUNDARY_VARIATION) {
			paraValue = ratingLowerBound + BOUNDARY_VARIATION;
		} // Of if

		if (paraValue > ratingUpperBound - BOUNDARY_VARIATION) {
			paraValue = ratingUpperBound - BOUNDARY_VARIATION;
		} // Of if

		double resultValue = -Math.log(
				Math.pow((ratingUpperBound - ratingLowerBound) / (paraValue - ratingLowerBound),
						parameterV) - 1);

		return resultValue;
	}// of glTransformJournal

	/**
	 ************************ 
	 * Transform the rating value back. The journal version.
	 * 
	 * @param paraValue
	 *            The given value.
	 ************************ 
	 */
	public double glInverseTransformJournal(double paraValue) {
		double resultValue = (ratingUpperBound - ratingLowerBound)
				/ Math.pow(Math.exp(-paraValue) + 1, 1 / parameterV) + ratingLowerBound;

		return resultValue;
	}// of glInverseTransformJournal

	/**
	 ************************ 
	 * Transform a rating value.
	 * 
	 * @param paraValue
	 *            The given value.
	 ************************ 
	 */
	public double glTransformConference(double paraValue) {
		if (paraValue > ratingUpperBound - BOUNDARY_VARIATION) {
			paraValue = ratingUpperBound - BOUNDARY_VARIATION;
		} else if (paraValue < ratingLowerBound + BOUNDARY_VARIATION) {
			paraValue = ratingLowerBound + BOUNDARY_VARIATION;
		} // of if

		double resultValue = -1 / parameterB
				* (Math.log(Math.pow(constant1 / (paraValue + constant2), parameterV) - 1));

		return resultValue;
	}// of glTransformConference

	/**
	 ************************ 
	 * @param paraX
	 * @return
	 ************************ 
	 */
	public double glInverseTransformConference(double paraValue) {
		double tempX = 0;
		double A = -10;
		double K = 10;
		double C = 1;
		double Q = 1;
		double B = 1000;
		double v = 0.01;

		tempX = A + (K - A) / (Math.pow((C + Q * Math.exp(-B * paraValue)), 1 / v));

		return tempX;
	}// of glInverseTransformConference

	/**
	 ************************ 
	 * Setter.
	 ************************ 
	 */
	public void setParameterV(double paraV) {
		parameterV = paraV;
	}// Of setParameterV

	/**
	 ************************ 
	 * Predict the rating of the user to the item
	 * 
	 * @param paraUser
	 *            The user index.
	 ************************ 
	 */
	public double predict(int paraUser, int paraItem) {
		double resultValue = 0;
		for (int i = 0; i < rank; i++) {
			resultValue += userSubspace[paraUser][i] * itemSubspace[paraItem][i];
		} // Of for i

		// Convert the prediction back.
		//System.out.print("converting back from " + resultValue);
		if (algorithm == GL_JOURNAL) {
			resultValue = glInverseTransformJournal(resultValue);
		} else {
			resultValue = glInverseTransformConference(resultValue);
		} // Of if
		//System.out.println(" to " + resultValue);

		return resultValue;
	}// Of predict

	/**
	 ************************ 
	 * Test the class.
	 ************************ 
	 */
	public static void glTransferTest(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			double paraLikeTreshold, boolean paraCompress, int paraRounds,
			int paraIncrementalRounds) {
		RatingSystem2DBoolean tempDataset = null;
		try {
			tempDataset = new RatingSystem2DBoolean(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound, paraLikeTreshold,
					paraCompress);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
		tempDataset.setAllTraining();

		MF2DBooleanIncrementalGLG tempLearner = new MF2DBooleanIncrementalGLG(tempDataset, 0);

		for (double tempValue = -10; tempValue <= 10; tempValue++) {
			double tempTransferedValue = tempLearner.glTransformJournal(tempValue);
			System.out.println("Value " + tempValue + ", transformed to " + tempTransferedValue);
		} // Of for

		for (double tempValue = -10; tempValue <= 10; tempValue++) {
			double tempTransferedValue = tempLearner.glInverseTransformJournal(tempValue);
			System.out.println(
					"Value " + tempValue + ", inverse transfered to " + tempTransferedValue);
		} // Of for
	}// Of glTransferTest

	/**
	 ************************ 
	 * The training testing scenario.
	 ************************ 
	 */
	public static void testIncremental(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			double paraLikeTreshold, boolean paraCompress, int paraRounds,
			int paraIncrementalRounds) {
		// Step 1. Read data and set parameters.

		RatingSystem2DBoolean tempDataset = null;
		try {
			tempDataset = new RatingSystem2DBoolean(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound, paraLikeTreshold,
					paraCompress);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
		tempDataset.setAllTraining();

		MF2DBooleanIncrementalGLG tempLearner = new MF2DBooleanIncrementalGLG(tempDataset, 0);

		tempLearner.setParameters(10, 0.0001, 0.005, NO_REGULAR, paraIncrementalRounds);

		// Step 2. Pre-train
		tempLearner.initializeSubspaces(0.5);
		System.out.println("Pre-training " + paraRounds + " rounds ...");
		tempLearner.train(paraRounds);

		// Step 3. Train for each user
		double tempMAE;
		int tempNumItemsForTrain = 0;
		int tempNumPredictions = 0;
		double tempErrorSum = 0;
		for (int i = 0; i < tempDataset.getNumUsers(); i++) {
			// System.out.println("User " + i);

			// Step 3.1 One half items, e.g., {0, 2, 4, ...} for training.
			tempNumItemsForTrain = tempDataset.getUserNumRatings(i) / 2;
			int[] tempIndices = new int[tempNumItemsForTrain];
			for (int j = 0; j < tempNumItemsForTrain; j++) {
				tempIndices[j] = tempDataset.getTriple(i, j * 2).item;
			} // Of for j
			tempDataset.setUserTraining(i, tempIndices);

			// Step 3.2 Incremental training.
			tempLearner.trainUser(i);

			// Step 3.3 Prediction and compute error.
			int tempItem;
			double tempPrediction;
			for (int j = tempNumItemsForTrain; j < tempDataset.getUserNumRatings(i); j++) {
				tempItem = tempDataset.getTriple(i, j).item;
				tempPrediction = tempLearner.predict(i, tempItem);
				tempErrorSum += Math.abs(tempPrediction - tempDataset.getTriple(i, j).rating);
				tempNumPredictions++;
			} // Of for j

			// Step 3.4 Restore data of this user.
			tempDataset.setUserAllTraining(i);

			// Step 3.5 Show message.
			tempMAE = tempErrorSum / tempNumPredictions;
			System.out.println(
					"MAE = " + tempErrorSum + " / " + tempNumPredictions + " = " + tempMAE);
		} // Of for i

		tempMAE = tempErrorSum / tempNumPredictions;
		System.out.println("With incremental updating, MAE = " + tempMAE);
	}// Of testIncremental

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		testIncremental("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, 0.5,
				false, 500, 100);
		glTransferTest("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10, 0.5,
				false, 500, 100);
	}// Of main
}// Of class MF2DBooleanIncremental
