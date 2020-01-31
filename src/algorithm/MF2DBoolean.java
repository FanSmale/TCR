package algorithm;

import java.io.*;

import datamodel.RatingSystem2DBoolean;

/**
 * Implement two matrix factorization algorithms. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCR.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: December 3, 2019.<br>
 *       Last modified: January 30, 2020.
 * @version 1.0
 */

public class MF2DBoolean extends RatingSystem2DBoolean {

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
	 * Regular scheme.
	 */
	int regularScheme;

	/**
	 * The range for the initial value of subspace values.
	 */
	double subspaceValueRange = 0.5;

	/**
	 * No regular scheme.
	 */
	public static final int NO_REGULAR = 0;

	/**
	 * No regular scheme.
	 */
	public static final int PQ_REGULAR = 1;

	/**
	 * Add the variation to the boundaries during data transferring to avoid
	 * NaN.
	 */
	public static final double BOUNDARY_VARIATION = 0.0001;

	/**
	 * How many rounds for training.
	 */
	int trainRounds = 200;

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
	 * @param paraRatingLowerBound
	 *            The lower bound of ratings.
	 * @param paraRatingUpperBound
	 *            The upper bound of ratings.
	 ************************ 
	 */
	public MF2DBoolean(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound) {
		super(paraFilename, paraNumUsers, paraNumItems, paraNumRatings, paraRatingLowerBound,
				paraRatingUpperBound);

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
	public void setParameters(int paraRank, double paraAlpha, double paraLambda,
			int paraRegularScheme, int paraTrainRounds) {
		rank = paraRank;
		alpha = paraAlpha;
		lambda = paraLambda;
		regularScheme = paraRegularScheme;
		trainRounds = paraTrainRounds;
	}// Of setParameters

	/**
	 ************************ 
	 * Get parameters.
	 ************************ 
	 */
	public String getParameters() {
		String resultString = "" + rank + ", " + alpha + ", " + lambda + ", " + regularScheme + ", "
				+ trainRounds;
		return resultString;
	}// Of setParameters

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
		subspaceValueRange = paraRange;
		userSubspace = new double[numUsers][rank];

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < rank; j++) {
				userSubspace[i][j] = (rand.nextDouble() - 0.5) * 2 * subspaceValueRange;
			} // of for j
		} // Of for i

		// SimpleTool.printMatrix(DataInfo.userFeature);
		itemSubspace = new double[numItems][rank];
		for (int i = 0; i < numItems; i++) {
			for (int j = 0; j < rank; j++) {
				itemSubspace[i][j] = (rand.nextDouble() - 0.5) * 2 * subspaceValueRange;
			} // Of for j
		} // Of for i
	}// Of initializeSubspaces

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
		return resultValue;
	}// Of predict

	/**
	 ************************ 
	 * Predict the ratings of the user to each item.
	 * 
	 * @param paraUser
	 *            The user index.
	 ************************ 
	 */
	public double[] predictForUser(int paraUser) {
		// System.out.println("predictForUser(" + paraUser + ")");
		double[] resultPredictions = new double[numItems];
		for (int i = 0; i < numItems; i++) {
			resultPredictions[i] = predict(paraUser, i);
		} // Of for i
		return resultPredictions;
	}// Of predictForUser

	/**
	 ************************ 
	 * Train.
	 * 
	 * @param paraRounds
	 *            The number of rounds.
	 ************************ 
	 */
	public void train() {
		train(trainRounds);
	}// Of train

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
				// System.out.println("MAE: " + mae());
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
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k]
							- lambda * userSubspace[tempUserId][k];

					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k]
							- lambda * itemSubspace[tempItemId][k];
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
	public static void testTrainingTesting(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			int paraRounds) {
		try {
			// Step 1. read the training and testing data
			MF2DBoolean tempMF = new MF2DBoolean(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound);

			tempMF.setParameters(10, 0.0001, 0.005, PQ_REGULAR, 200);
			tempMF.initializeTraining(0.8);
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
	 * Training and testing using the same data.
	 ************************ 
	 */
	public static void testAllTrainingTesting(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings, double paraRatingLowerBound,
			double paraRatingUpperBound, int paraRounds) {
		try {
			// Step 1. read the training and testing data
			MF2DBoolean tempMF = new MF2DBoolean(paraFilename, paraNumUsers, paraNumItems,
					paraNumRatings, paraRatingLowerBound, paraRatingUpperBound);

			tempMF.setParameters(10, 0.0001, 0.005, NO_REGULAR, 200);
			tempMF.initializeTraining(1.0);
			tempMF.adjustUsingMeanRating();

			// tempMF.setTestingSetRemainder(2);
			// Step 2. Initialize the feature matrices U and V
			tempMF.initializeSubspaces(0.5);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");

			tempMF.train(paraRounds);

			tempMF.initializeTraining(0);
			double tempMAE = tempMF.mae();
			double tempRSME = tempMF.rsme();
			System.out.println("Finally, MAE = " + tempMAE + ", RSME = " + tempRSME);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}// Of testAllTrainingTesting

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		testAllTrainingTesting("data/jester-data-1/jester-data-1.txt", 24983, 101, 1810455, -10, 10,
				200);
		// testSameTrainingTesting("data/jester-data-1/jester-data-1.txt",
		// 24983, 101, 1810455, -10, 10, 500);
	}// Of main
}// Of class MatrixFactorization2DBooleanIndication
