package algorithm;

import java.io.*;
import java.util.Random;
import datamodel.Triple;

/*
 * @(#)MatrixFactorization.java
 * 
 * Project: Matrix factorization for recommender systems.
 * Author: Fan Min, Yuan-Yuan Xu
 * www.fansmale.com
 * Email: minfan@swpu.edu.cn, minfanphd@163.com.
 * Created: December 3, 2019.
 * Last modified: Jan 16, 2020.
 */

public class MatrixFactorization {
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
	 * Number of ratings in the training set.
	 */
	int numTrainingRatings;

	/**
	 * Number of ratings in the testing set.
	 */
	int numTestingRatings;

	/**
	 * Training data.
	 */
	Triple[] trainingData;

	/**
	 * Testing data.
	 */
	Triple[] testingData;

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
	 * The number of items rated by each user.
	 */
	int[] userRates;

	/**
	 * The offset of each user in the source data.
	 */
	int[] userOffset;

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
	public MatrixFactorization(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings, double paraRatingLowerBound,
			double paraRatingUpperBound, double paraTrainingFraction) {
		numUsers = paraNumUsers;
		numItems = paraNumItems;
		numRatings = paraNumRatings;
		ratingLowerBound = paraRatingLowerBound;
		ratingUpperBound = paraRatingUpperBound;

		// data = new Triple[numRatings];
		try {
			readData(paraFilename, paraTrainingFraction);
		} catch (Exception ee) {
			System.out.println("File " + paraFilename + " cannot be read! "
					+ ee);
			System.exit(0);
		}// Of try

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
	public void setParameters(int paraRank, double paraAlpha,
			double paraLambda, int paraRegularScheme) {
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
	public static Triple[] readData(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings) throws IOException {
		File file = new File(paraFilename);
		BufferedReader buffRead = new BufferedReader(new InputStreamReader(
				new FileInputStream(file)));

		// Allocate space.
		Triple[] resultData = new Triple[paraNumRatings];

		int tempUserIndex = 0;
		int tempIndex = 0;
		while (buffRead.ready()) {
			String str = buffRead.readLine();
			String[] parts = str.split(SPLIT_SIGN);

			for (int i = 1; i < paraNumItems; i++) {
				int tempItemIndex = i - 1;// item id
				double tempRating = Double.parseDouble(parts[i]);// rating

				if (tempRating != 99) {
					resultData[tempIndex] = new Triple(tempUserIndex, tempItemIndex, tempRating);
					tempIndex++;
				}// Of if
			}// Of for i
			tempUserIndex++;
		} // Of while
		buffRead.close();

		return resultData;
	}// Of readData

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @param paraTrainingFraction
	 *            The fraction of the training set.
	 * @throws IOException
	 ************************ 
	 */
	public void readData(String paraFilename, double paraTrainingFraction)
			throws IOException {
		// Step 1. Read all data.
		Triple[] tempAllData = readData(paraFilename, numUsers, numItems,
				numRatings);

		numTrainingRatings = (int) (numRatings * paraTrainingFraction);
		numTestingRatings = numRatings - numTrainingRatings;
		System.out.println("numTrainingRatings = " + numTrainingRatings);
		System.out.println("numTestingRatings = " + numTestingRatings);

		trainingData = new Triple[numTrainingRatings];
		testingData = new Triple[numTestingRatings];

		// Step 2. Split to training and testing sets.
		int tempTrainingIndex = 0;
		int tempTestingIndex = 0;
		double tempDouble;
		boolean tempTrain = false;
		for (int i = 0; i < tempAllData.length; i++) {
			tempDouble = rand.nextDouble();
			if (tempDouble <= paraTrainingFraction) {
				if (tempTrainingIndex < numTrainingRatings) {
					tempTrain = true;
				} else {
					tempTrain = false;
				}// Of if
			} else {
				if (tempTestingIndex < numTestingRatings) {
					tempTrain = false;
				} else {
					tempTrain = true;
				}// Of if
			}// Of if

			if (tempTrain) {
				trainingData[tempTrainingIndex] = tempAllData[i];
				tempTrainingIndex++;
			} else {
				testingData[tempTestingIndex] = tempAllData[i];
				tempTestingIndex++;
			}// Of if
		}// Of for i

		// Step 3. Compute meanRating
		adjustUsingMeanRating();
	}// Of readData

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void readTrainingData(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings) throws IOException {
		// Step 1. Read data.
		trainingData = readData(paraFilename, paraNumUsers, paraNumItems,
				paraNumRatings);

		// Step 2. Set the mean rating.
		adjustUsingMeanRating();
	}// Of readTrainingData

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void adjustUsingMeanRating() {
		double tempRatingSum = 0;
		for (int i = 0; i < trainingData.length; i++) {
			tempRatingSum += trainingData[i].rating;
		}// Of for i
		meanRating = tempRatingSum / trainingData.length;

		// Step 3. Update the ratings in the training set.
		for (int i = 0; i < trainingData.length; i++) {
			trainingData[i].rating -= meanRating;
		}// Of for i
		
		// Step 4. Also update the bounds.
		ratingLowerBound -= meanRating;
		ratingUpperBound -= meanRating;
	}// Of adjustUsingMeanRating

	/**
	 ************************ 
	 * Read the data from the file.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @throws IOException
	 ************************ 
	 */
	public void readTestingData(String paraFilename, int paraNumUsers,
			int paraNumItems, int paraNumRatings) throws IOException {
		// Step 1. Read data.
		testingData = readData(paraFilename, paraNumUsers, paraNumItems,
				paraNumRatings);

		// Step 3. Update the ratings in the training set.
		for (int i = 0; i < testingData.length; i++) {
			testingData[i].rating -= meanRating;
		}// Of for i
	}// Of readTestingData

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
			// User����������Item�������ĳ˻�. ���ִ洢��ʽ�ǳ�����
			resultValue += userSubspace[paraUser][i]
					* itemSubspace[paraItem][i];
		} // of for i
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
				System.out.println("MAE: " + mae());
			}// Of if
		}// Of for i
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
		for (int i = 0; i < numTrainingRatings; i++) {
			int tempUserId = (int) trainingData[i].user;
			int tempItemId = (int) trainingData[i].item;
			double tempRate = (double) trainingData[i].rating;

			double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
			// tempResidual = Math.abs(tempResidual);

			// Update user subspace
			double tempValue = 0;
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * itemSubspace[tempItemId][j];
				userSubspace[tempUserId][j] += alpha * tempValue;
			}// Of for j

			// Update item subspace
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * userSubspace[tempUserId][j];

				itemSubspace[tempItemId][j] += alpha * tempValue;
			}// Of for j
		}// Of for i
	}// Of updateNoRegular

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void updatePQRegular() {
		for (int i = 0; i < numTrainingRatings; i++) {
			int tempUserId = (int) trainingData[i].user;
			int tempItemId = (int) trainingData[i].item;
			double tempRate = (double) trainingData[i].rating;

			double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
			// tempResidual = Math.abs(tempResidual);

			// Update user subspace
			double tempValue = 0;
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * itemSubspace[tempItemId][j]
						- lambda * userSubspace[tempUserId][j];

				userSubspace[tempUserId][j] += alpha * tempValue;
			}// Of for j

			// Update item subspace
			for (int j = 0; j < rank; j++) {
				tempValue = 2 * tempResidual * userSubspace[tempUserId][j]
						- lambda * itemSubspace[tempItemId][j];
				itemSubspace[tempItemId][j] += alpha * tempValue;
			}// Of for j
		}// Of for i
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

		for (int i = 0; i < numTestingRatings; i++) {
			int tempUserIndex = testingData[i].user;
			int tempItemIndex = testingData[i].item;
			double tempRate = testingData[i].rating;

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

		for (int i = 0; i < numTestingRatings; i++) {
			int tempUserIndex = testingData[i].user;
			int tempItemIndex = testingData[i].item;
			double tempRate = testingData[i].rating;

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
		} // Of for i

		return (resultMae / tempTestCount);
	}// Of mae

	/**
	 ************************ 
	 * Compute the MAE.
	 * 
	 * @return MAE of the current factorization.
	 ************************ 
	 */
	public static void testTrainingTesting(String paraFilename,
			int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound,
			int paraRounds) {
		try {
			// Step 1. read the training and testing data
			MatrixFactorization tempMF = new MatrixFactorization(paraFilename,
					paraNumUsers, paraNumItems, paraNumRatings,
					paraRatingLowerBound, paraRatingUpperBound, 0.9);

			tempMF.setParameters(5, 0.0001, 0.005, PQ_REGULAR);
			// tempMF.setTestingSetRemainder(2);
			// Step 2. Initialize the feature matrices U and V
			tempMF.initializeSubspaces(0.5);

			// Step 3. update and predict
			System.out.println("Begin Training ! ! !");

			tempMF.train(paraRounds);

			double tempMAE = tempMF.mae();
			double tempRSME = tempMF.rsme();
			System.out.println("Finally, MAE = " + tempMAE + ", RSME = "
					+ tempRSME);
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
		testTrainingTesting("data/jester-data-1/jester-data-1.txt", 24983, 101,
				1810455, -10, 10, 200);
	}// Of main
}// Of class MatrixFactorization
