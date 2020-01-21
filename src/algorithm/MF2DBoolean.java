package algorithm;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import datamodel.Triple;

/*
 * @(#)MF2DBoolean.java
 
 * Project: Matrix factorization for recommender systems.
 * The data is organized in 2D to enable incremental learning. 
 * Author: Fan Min, Yuan-Yuan Xu
 * www.fansmale.com
 * Email: minfan@swpu.edu.cn, minfanphd@163.com.
 * Created: December 3, 2019.
 * Last modified: Jan 19, 2020.
 */

public class MF2DBoolean {
	/**
	 * A sign to help reading the data file.
	 */
	public static final String SPLIT_SIGN = new String("	");

	/**
	 * Used to generate random numbers.
	 */
	Random rand = new Random();

	/**
	 * Number of users.
	 */
	int numUsers;

	/**
	 * Number of items.
	 */
	int numItems;

	/**
	 * Number of ratings.
	 */
	int numRatings;

	/**
	 * The whole data.
	 */
	Triple[][] data;

	/**
	 * Which elements belong to the training set.
	 */
	boolean[][] trainingIndicationMatrix;

	/**
	 * Mean rating calculated from the training sets.
	 */
	double meanRating;

	/**
	 * A parameter for controlling learning regular.
	 */
	double alpha;

	/**
	 * A parameter for controlling the learning speed.
	 */
	double lambda;

	/**
	 * The low rank of the small matrices.
	 */
	int rank;

	/**
	 * The user matrix U.
	 */
	double[][] userSubspace;

	/**
	 * The item matrix V.
	 */
	double[][] itemSubspace;

	/**
	 * The lower bound of the rating value.
	 */
	double ratingLowerBound;

	/**
	 * The upper bound of the rating value.
	 */
	double ratingUpperBound;

	/**
	 * Regular scheme.
	 */
	int regularScheme;

	/**
	 * No regular scheme.
	 */
	public static final int NO_REGULAR = 0;

	/**
	 * No regular scheme.
	 */
	public static final int PQ_REGULAR = 1;

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
	public MF2DBoolean(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound) {
		numUsers = paraNumUsers;
		numItems = paraNumItems;
		numRatings = paraNumRatings;
		ratingLowerBound = paraRatingLowerBound;
		ratingUpperBound = paraRatingUpperBound;

		// data = new Triple[numRatings];
		try {
			readData(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
		} catch (Exception ee) {
			System.out.println("File " + paraFilename + " cannot be read! " + ee);
			System.exit(0);
		} // Of try

		initialize();
	}// Of the first constructor

	/**
	 ************************ 
	 * Set parameters.
	 * 
	 * @param paraRank
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void setParameters(int paraRank, double paraAlpha, double paraLambda, int paraRegularScheme) {
		rank = paraRank;
		alpha = paraAlpha;
		lambda = paraLambda;
		regularScheme = paraRegularScheme;
	}// Of setParameters

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public Triple[][] readData(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings)
			throws IOException {
		File file = new File(paraFilename);
		BufferedReader buffRead = new BufferedReader(new InputStreamReader(new FileInputStream(file)));

		// Allocate space.
		data = new Triple[paraNumUsers][];
		trainingIndicationMatrix = new boolean[numUsers][];

		Triple[] tempTripleArrayForUser = new Triple[paraNumItems];
		int tempCurrentUserRatings = 0;

		int tempUserIndex = 0;
		while (buffRead.ready()) {
			String str = buffRead.readLine();
			String[] parts = str.split(SPLIT_SIGN);

			// The first loop to read the current line for one user.
			tempCurrentUserRatings = 0;
			for (int i = 1; i < paraNumItems; i++) {
				int tempItemIndex = i - 1;// item id
				double tempRating = Double.parseDouble(parts[i]);// rating

				if (tempRating != 99) {
					tempTripleArrayForUser[tempCurrentUserRatings] = new Triple(tempUserIndex, tempItemIndex,
							tempRating);
					tempCurrentUserRatings++;
				} // Of if
			} // Of for i

			// The second loop to copy.
			data[tempUserIndex] = new Triple[tempCurrentUserRatings];
			trainingIndicationMatrix[tempUserIndex] = new boolean[tempCurrentUserRatings];
			for (int i = 0; i < tempCurrentUserRatings; i++) {
				data[tempUserIndex][i] = tempTripleArrayForUser[i];
			} // Of for i
			tempUserIndex++;
		} // Of while
		buffRead.close();

		return data;
	}// Of readData

	/**
	 ************************ 
	 * Set the training part.
	 * 
	 * @param paraTrainingFraction
	 *            The fraction of the training set.
	 * @throws IOException
	 ************************ 
	 */
	public void initializeTraining(double paraTrainingFraction) {
		// Step 1. Read all data.
		int tempTotalTrainingSize = (int) (numRatings * paraTrainingFraction);
		int tempTotalTestingSize = numRatings - tempTotalTrainingSize;

		int tempTrainingSize = 0;
		int tempTestingSize = 0;
		double tempDouble;

		// Step 2. Handle each user.
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				tempDouble = rand.nextDouble();
				if (tempDouble <= paraTrainingFraction) {
					if (tempTrainingSize < tempTotalTrainingSize) {
						trainingIndicationMatrix[i][j] = true;
						tempTrainingSize++;
					} else {
						trainingIndicationMatrix[i][j] = false;
						tempTestingSize++;
					} // Of if
				} else {
					if (tempTestingSize < tempTotalTestingSize) {
						trainingIndicationMatrix[i][j] = false;
						tempTestingSize++;
					} else {
						trainingIndicationMatrix[i][j] = true;
						tempTrainingSize++;
					} // Of if
				} // Of if
			} // Of for j
		} // Of for i

		System.out.println("" + tempTrainingSize + " training instances.");
		System.out.println("" + tempTestingSize + " testing instances.");
	}// Of initializeTraining

	/**
	 ************************ 
	 * Set all data for training.
	 ************************ 
	 */
	public void setAllTraining() {
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				trainingIndicationMatrix[i][j] = true;
			} // Of for j
		} // Of for i
	}// Of setAllTraining



	/**
	 ************************ 
	 * Adjust the training data with the mean rating. The ratings are subtracted
	 * with the mean rating. So do the rating bounds.
	 ************************ 
	 */
	public void adjustUsingMeanRating() {
		// Step 1. Calculate the mean rating.
		double tempRatingSum = 0;
		int tempTrainingSize = 0;
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				if (trainingIndicationMatrix[i][j]) {
					tempRatingSum += data[i][j].rating;
					tempTrainingSize++;
				} // Of if
			} // Of for j
		} // Of for i
		meanRating = tempRatingSum / tempTrainingSize;

		// Step 2. Update the ratings in the training set.
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingIndicationMatrix[i].length; j++) {
				if (trainingIndicationMatrix[i][j]) {
					data[i][j].rating -= meanRating;
				} // Of if
			} // Of for j
		} // Of for i

		// Step 3. Also update the bounds.
		ratingLowerBound -= meanRating;
		ratingUpperBound -= meanRating;
	}// Of adjustUsingMeanRating

	/**
	 ************************ 
	 * Initialize some variables.
	 ************************ 
	 */
	void initialize() {
		rank = 5;
		alpha = 0.0001;
		lambda = 0.005;
		regularScheme = 0;
	}// Of initialize

	/**
	 ************************ 
	 * Initialize subspaces. Each value is in [-paraRange, +paraRange].
	 * 
	 * @paraRange The range of the initial values.
	 ************************ 
	 */
	void initializeSubspaces(double paraRange) {
		userSubspace = new double[numUsers][rank];

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < rank; j++) {
				userSubspace[i][j] = (rand.nextDouble() - 0.5) * 2 * paraRange;
			} // of for j
		} // Of for i

		// SimpleTool.printMatrix(DataInfo.userFeature);
		itemSubspace = new double[numItems][rank];
		for (int i = 0; i < numItems; i++) {
			for (int j = 0; j < rank; j++) {
				itemSubspace[i][j] = 1 * (rand.nextDouble() - 0.5) * 2 * paraRange;
			} // Of for j
		} // Of for i
	}// of initializeSubspaces

	/**
	 ************************ 
	 * Predict the rating of the user to the item
	 * 
	 * @param paraUser
	 *            The user index.
	 ************************ 
	 */
	public double predict(int paraUser, int paraItem) {
		// System.out.println("Predict in the superclass");
		double resultValue = 0;
		for (int i = 0; i < rank; i++) {
			resultValue += userSubspace[paraUser][i] * itemSubspace[paraItem][i];
		} // Of for i
		return resultValue;
	}// Of predict

	/**
	 ************************ 
	 * Train.
	 * 
	 * @param paraRounds
	 *            The number of rounds.
	 ************************ 
	 */
	public void train(int paraRounds) {
		for (int i = 0; i < paraRounds; i++) {
			update();
			if (i % 50 == 0) {
				// Show the process
				System.out.println("Round " + i);
				//System.out.println("MAE: " + mae());
			} // Of if
		} // Of for i
	}// Of train



	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void update() {
		switch (regularScheme) {
		case NO_REGULAR:
			updateNoRegular();
			break;
		case PQ_REGULAR:
			updatePQRegular();
			break;
		default:
			System.out.println("Unsupported regular scheme: " + regularScheme);
			System.exit(0);
		}// Of switch
	}// Of update


	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updateNoRegular() {
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < data[i].length; j++) {
				// Ignore the testing set.
				if (!trainingIndicationMatrix[i][j]) {
					continue;
				} // Of if

				int tempUserId = data[i][j].user;
				int tempItemId = data[i][j].item;
				double tempRate = data[i][j].rating;

				double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
				// tempResidual = Math.abs(tempResidual);

				// Update user subspace
				double tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k];
					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k];

					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of updateNoRegular


	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updatePQRegular() {
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < data[i].length; j++) {
				// Ignore the testing set.
				if (!trainingIndicationMatrix[i][j]) {
					continue;
				} // Of if

				int tempUserId = data[i][j].user;
				int tempItemId = data[i][j].item;
				double tempRate = data[i][j].rating;

				double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
				// tempResidual = Math.abs(tempResidual);

				// Update user subspace
				double tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k] - lambda * userSubspace[tempUserId][k];

					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k] - lambda * itemSubspace[tempItemId][k];
					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of updatePQRegular



	/**
	 ************************ 
	 * Compute the RSME.
	 * 
	 * @return RSME of the current factorization.
	 ************************ 
	 */
	public double rsme() {
		double resultRsme = 0;
		int tempTestCount = 0;

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < data[i].length; j++) {
				// Ignore the training set.
				if (trainingIndicationMatrix[i][j]) {
					continue;
				} // Of if

				int tempUserIndex = data[i][j].user;
				int tempItemIndex = data[i][j].item;
				double tempRate = data[i][j].rating;

				double tempPrediction = predict(tempUserIndex, tempItemIndex);// +
																				// DataInfo.mean_rating;

				if (tempPrediction < ratingLowerBound) {
					tempPrediction = ratingLowerBound;
				} else if (tempPrediction > ratingUpperBound) {
					tempPrediction = ratingUpperBound;
				} // Of if

				double tempError = tempRate - tempPrediction;
				resultRsme += tempError * tempError;
				tempTestCount++;
			} // Of for j
		} // Of for i

		return Math.sqrt(resultRsme / tempTestCount);
	}// Of rsme

	/**
	 ************************ 
	 * Compute the MAE.
	 * 
	 * @return MAE of the current factorization.
	 ************************ 
	 */
	public double mae() {
		double resultMae = 0;
		int tempTestCount = 0;

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < data[i].length; j++) {
				// Ignore the training set.
				if (trainingIndicationMatrix[i][j]) {
					continue;
				} // Of if

				int tempUserIndex = data[i][j].user;
				int tempItemIndex = data[i][j].item;
				double tempRate = data[i][j].rating;

				double tempPrediction = predict(tempUserIndex, tempItemIndex);

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
			} // Of for j
		} // Of for i

		return (resultMae / tempTestCount);
	}// Of mae

	/**
	 ************************ 
	 * The training testing scenario.
	 ************************ 
	 */
	public static void testTrainingTesting(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, int paraRounds) {
		try {
			// Step 1. read the training and testing data
			MF2DBoolean tempMF = new MF2DBoolean(paraFilename, paraNumUsers, paraNumItems, paraNumRatings,
					paraRatingLowerBound, paraRatingUpperBound);

			tempMF.setParameters(10, 0.0001, 0.005, PQ_REGULAR);
			tempMF.initializeTraining(0.9);
			tempMF.adjustUsingMeanRating();

			// tempMF.setTestingSetRemainder(2);
			// Step 2. Initialize the feature matrices U and V
			tempMF.initializeSubspaces(0.5);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");

			tempMF.train(paraRounds);

			double tempMAE = tempMF.mae();
			double tempRSME = tempMF.rsme();
			System.out.println("Finally, MAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of testTrainingTesting

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		testTrainingTesting("data/jester-data-1/jester-data-1.txt", 24983,
		101, 1810455, -10, 10, 300);
		// testSameTrainingTesting("data/jester-data-1/jester-data-1.txt",
		// 24983, 101, 1810455, -10, 10, 500);
	}// Of main
}// Of class MatrixFactorization2DBooleanIndication
