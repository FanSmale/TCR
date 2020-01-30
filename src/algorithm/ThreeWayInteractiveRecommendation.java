package algorithm;

import algorithm.GLFactorizationJournal;
import java.io.*;
import java.util.Arrays;
import java.util.Random;
import datamodel.*;
/**
 * ThreeWayInteractiveRecommendation.java
 * Project: Matrix factorization for recommender systems.
 * The data is organized in 2D of triples. Boolean means that a boolean matrix indicates 
 * the training set. The purpose is to enable incremental learning. Now only uncompressed 
 * data file is supported, that is, missing value is indicated by 99. In the near future, 
 * the data organized by triples should also be supported. <br>
 * @author Fan Min
 * www.fansmale.com, github.com/fansmale
 * Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created July 25, 2019.
 * Last modified: January 27, 2020.
 */
public class ThreeWayInteractiveRecommendation {
	GLFactorizationJournal tempGLF;
	/**
	 * The split sign in datasets.
	 */
	public static final String SPLIT_SIGN = new String("\t");

	/**
	 * The number of users.
	 */
	int numUsers;

	/**
	 * The number of items.
	 */
	int numItems;

	/**
	 * The number of ratings.
	 */
	int numRatings;

	/**
	 * The user vector.
	 */
	int[] userVector;

	/**
	 * The item vector.
	 */
	int[] itemVector;

	/**
	 * The rating vector.
	 */
	double[] ratingVector;

	/**
	 * The starting point of the user in the vector. The length should be (m + 1),
	 * where the last one is numRatings. In this way we do not have to store how
	 * many items have be rated by each user.
	 */
	int[] userStartingPoints;

	/**
	 * The number of items rated by the corresponding users.
	 */
	int[] userCount;

	/**
	 * The rating times of each item.
	 */
	int[] itemPopArray;

	/**
	 * The sum of ratings of each item.
	 */
	double[] itemRatingSumArray;

	/**
	 * The average ratings of each item.
	 */
	double[] itemAverageRatingArray;

	/**
	 * The cost matrix.
	 */
	double[][] costMatrix;

	/**
	 * The maturity value array.
	 */
	double[] maturityValueArray;

	/**
	 * The default recommendation list length.
	 */
	public static final int DEFAULT_RECOMMENDATION_LENGTH = 10;

	/**
	 * The recommendation list length for each round.
	 */
	int recommendationLength;

	/**
	 * Some items are recommended, while others are promoted.
	 */
	double recommendRatio;

	/**
	 * The default recommendation ratio.
	 */
	public static final double DEFAULT_RECOMMENDATION_RATIO = 0.3;

	/**
	 * Non-recommend.
	 */
	public static final int NON_RECOMMEND = 0;

	/**
	 * Promote.
	 */
	public static final int PROMOTE = 1;

	/**
	 * Recommend.
	 */
	public static final int RECOMMEND = 2;

	/**
	 * The default like threshold.
	 */
	public static final int DEFAULT_LIKE_THRESHOLD = 3;

	/**
	 *  
	 * Add the variation to the boundaries during data transferring to avoid
	 * NaN.
	 */
	public static final double BOUNDARY_VARIATION = 0.0001;
	/**
	 * The like threshold.
	 */
	double likeThreshold;

	/**
	 * The default maturity threshold.
	 */
	public static final double DEFAULT_MATURITY_THRESHOLD = 1000;

	/**
	 * The maturity threshold.
	 */
	double maturityThreshold;

	/**
	 * The default maturity value for each item.
	 */
	public static final double DEFAULT_MATURITY_FOR_EACH_ITEM = 3;

	/**
	 * The maturity value for each item.
	 */
	double maturityForEachItem;

	/**
	 * The final recommendation behavior. 0 stands for not recommend. 1 stands for
	 * promote. 2 stands for recommend. The length is the same as the number of
	 * ratings in the original data.
	 */
	int[] recommendationBehavior;

	/**
	 * The recommendation list for the current user.
	 */
	boolean[] currentUserRecommendationList;

	/**
	 * The promotion list for the current user.
	 */
	boolean[] currentUserPromotionList;

	/**
	 * The actual ratings of the current user.
	 */
	double[] currentUserActualRatings;

	/**
	 * The maximum of item pop.
	 */
	int maxItemPop;

	/**
	 * The default item pop threshold. public static final double
	 * DEFAULT_POP_THRESHOLD = 0.5;
	 */

	/**
	 * The item pop threshold (normalization).
	 */
	double popThreshold = 0.5;

	/**
	 * The threshold for recommendation (predicted by M-distance).
	 */
	double recommendScoreThreshold = 3.8;

	/**
	 * The threshold for promotion (predicted by M-distance).
	 */
	double promoteScoreThreshold = 3.2;

	/**
	 * The popular items (for recommend).
	 */
	int[] popItems;

	/**
	 * The item semi-pop threshold (normalization).
	 */
	double semiPopThreshold = 0.3;

	/**
	 * The semi-popular items (for recommend).
	 */
	int[] semiPopItems;

	/**
	 * Recommendation summary
	 */
	int[][] recommendationSummaryMatrix;

	/**
	 * Neighborhood radius.
	 */
	double neighborhoodRadius;

	/**
	 * ratingLowerBound
	 */
	double ratingLowerBound = -10;
	/**
	 * ratingUpperBound
	 */
	double ratingUpperBound = 10;
	/**
	 *********************************** 
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The rating filename.
	 * @param paraUsers
	 *            The number of users.
	 * @param paraItems
	 *            The number of items.
	 * @throws IOException
	 * @throws NumberFormatException
	 *********************************** 
	 */
	public ThreeWayInteractiveRecommendation(String paraFilename, int paraUsers, int paraItems, int paraRatings)
			throws NumberFormatException, IOException {

		numUsers = paraUsers;
		numItems = paraItems;
		numRatings = paraRatings;

		// Step 1. Allocate space.
		userVector = new int[numRatings];
		itemVector = new int[numRatings];
		ratingVector = new double[numRatings];

		userCount = new int[numUsers];
		userStartingPoints = new int[numUsers + 1];
		userStartingPoints[0] = 0;
		userStartingPoints[numUsers] = numRatings;

		itemPopArray = new int[numItems];
		itemRatingSumArray = new double[numItems];
		recommendationBehavior = new int[numRatings];

		currentUserRecommendationList = new boolean[numItems];
		currentUserPromotionList = new boolean[numItems];
		currentUserActualRatings = new double[numItems];

		recommendationSummaryMatrix = new int[numUsers][6];

		maturityValueArray = new double[4];
		maturityValueArray[0] = 5;
		maturityValueArray[1] = 2.5;
		maturityValueArray[2] = 4;
		maturityValueArray[3] = 2;

		neighborhoodRadius = 1.3;

		
		// Step 2. Read data from file.
		// Q.
		File tempFile = new File(paraFilename);
		BufferedReader tempBuffReader = new BufferedReader(new InputStreamReader(new FileInputStream(tempFile)));

		int tempIndex = 0;
		String tempLine;
		String[] tempParts;
		int tempUser, tempItem;
		double tempRating;
		while (tempBuffReader.ready()) {
			tempLine = tempBuffReader.readLine();
			tempParts = tempLine.split(SPLIT_SIGN);

			// int user = Integer.parseInt(parts[0]) ;// user id
			// int item = Integer.parseInt(parts[1]) ;// item id
			// double rating = Double.parseDouble(parts[2]);// rating
			tempUser = Integer.parseInt(tempParts[0]) - 1;// user id
			tempItem = Integer.parseInt(tempParts[1]) - 1;// item id
			tempRating = Double.parseDouble(tempParts[2]);// rating

			userCount[tempUser]++;
			itemPopArray[tempItem]++;
			itemRatingSumArray[tempItem] += tempRating;

			userVector[tempIndex] = tempUser;
			itemVector[tempIndex] = tempItem;
			ratingVector[tempIndex] = tempRating;

			tempIndex++;
		} // Of while
		tempBuffReader.close();
		itemAverageRatingArray = new double[numItems];
		for (int i = 0; i < numItems; i++) {
			itemAverageRatingArray[i] = itemRatingSumArray[i] / itemPopArray[i];
		} // Of for i

		// Step 3. Initialize.
		initialize();
	}// Of the constructor

	/**
	 *********************************** 
	 * Initialize.
	 *********************************** 
	 */
	private void initialize() {
		recommendationLength = DEFAULT_RECOMMENDATION_LENGTH;
		recommendRatio = DEFAULT_RECOMMENDATION_RATIO;

		costMatrix = new double[3][2];
		costMatrix[0][0] = 2; // NN
		costMatrix[0][1] = 40; // NP
		costMatrix[1][0] = 20; // BN
		costMatrix[1][1] = 10; // BP
		costMatrix[2][0] = 50; // PN
		costMatrix[2][1] = 6; // PP

		likeThreshold = DEFAULT_LIKE_THRESHOLD;
		maturityThreshold = DEFAULT_MATURITY_THRESHOLD;
		maturityForEachItem = DEFAULT_MATURITY_FOR_EACH_ITEM;

		setUserStartingPoints();
		maxItemPop = computeMaxItemPop();

		popItems = null;
		semiPopItems = null;
	}// Of initialize

	/**
	 *********************************** 
	 * Set the maturity value array.
	 * 
	 * @param paraArray
	 *            The maturity value array with four values.
	 *********************************** 
	 */
	public void setMaturityValueArray(double[] paraArray) {
		maturityValueArray = paraArray;
	}// Of setMaturityValueArray

	/**
	 *********************************** 
	 * Compute both popular items, which may be recommended, and semi-popular items,
	 * which may be promoted.
	 * 
	 *********************************** 
	 */
	public void computePopAndSemipopItems(double paraPopThreshold, double paraSemiPopThreshold) {
		// Step 1. Compute popular items.
		int tempNumPopItems = 0;
		// Step 1.1 Compute the length
		for (int i = 0; i < itemPopArray.length; i++) {
			if (itemPopArray[i] > maxItemPop * paraPopThreshold) {
				tempNumPopItems++;
			} // Of if
		} // Of for i

		// Step 1.2 Compute the array
		popItems = new int[tempNumPopItems];
		tempNumPopItems = 0;
		for (int i = 0; i < itemPopArray.length; i++) {
			if (itemPopArray[i] > maxItemPop * paraPopThreshold) {
				popItems[tempNumPopItems] = i;
				tempNumPopItems++;
			} // Of if
		} // Of for i

		// Step 2. Compute semi-popular items.
		int tempNumSemiPopItems = 0;
		// Step 2.1 Compute the length
		for (int i = 0; i < itemPopArray.length; i++) {
			if ((itemPopArray[i] > maxItemPop * paraSemiPopThreshold)
					&& (itemPopArray[i] <= maxItemPop * paraPopThreshold)) {
				tempNumSemiPopItems++;
			} // Of if
		} // Of for i

		// Step 1.2 Compute the array
		semiPopItems = new int[tempNumSemiPopItems];
		tempNumSemiPopItems = 0;
		for (int i = 0; i < itemPopArray.length; i++) {
			if ((itemPopArray[i] > maxItemPop * paraSemiPopThreshold)
					&& (itemPopArray[i] <= maxItemPop * paraPopThreshold)) {
				semiPopItems[tempNumSemiPopItems] = i;
				tempNumSemiPopItems++;
			} // Of if
		} // Of for i

		System.out.println("Pop items: " + Arrays.toString(popItems));
		System.out.println("Semi-popular items: " + Arrays.toString(semiPopItems));
	}// Of computePopAndSemipopItems

	/**
	 *********************************** 
	 * Set the userStartingPoints.
	 *********************************** 
	 */
	public void setUserStartingPoints() {
		for (int i = 1; i <= numUsers; i++) {
			userStartingPoints[i] = userStartingPoints[i - 1] + userCount[i - 1];
		} // Of for
	}// Of setUserStartingPoints

	/**
	 *********************************** 
	 * Set popular and semi-popular thresholds.
	 *********************************** 
	 */
	public void setPopThresholds(double paraPopThreshold, double paraSemiPopThreshold) {
		popThreshold = paraPopThreshold;
		semiPopThreshold = paraSemiPopThreshold;
	}// Of setPopThresholds

	/**
	 *********************************** 
	 * Compute the maximum Of item pop.
	 *********************************** 
	 */
	public int computeMaxItemPop() {
		int resuleMaxPop = itemPopArray[0];
		for (int i = 1; i < itemPopArray.length; i++) {
			if (resuleMaxPop < itemPopArray[i]) {
				resuleMaxPop = itemPopArray[i];
			} // Of if
		} // Of for i
		return resuleMaxPop;
	}// Of computeMaxItemPop

	/**
	 *********************************** 
	 * Set the recommendation length.
	 *********************************** 
	 */
	public void setRecommendationLength(int paraLength) {
		recommendationLength = paraLength;
	}// Of setRecommendationLength

	/**
	 *********************************** 
	 * Set the cost matrix.
	 *********************************** 
	 */
	public void setCostMatrix(int paraNN, int paraNP, int paraBN, int paraBP, int paraPN, int paraPP) {
		costMatrix[0][0] = paraNN;
		costMatrix[0][1] = paraNP;
		costMatrix[1][0] = paraBN;
		costMatrix[1][1] = paraBP;
		costMatrix[2][0] = paraPN;
		costMatrix[2][1] = paraPP;
		// Q.
	}// Of setCostMatrix

	/**
	 *********************************** 
	 * Compute the total cost.
	 *********************************** 
	 */
	public double computeTotalCost() {
		// Compare ratingVector with recommendationBehavior.
		double resultTotalcost = 0;
		int tempBehavior;
		int tempLike;
		for (int i = 0; i < numRatings; i++) {
			tempBehavior = recommendationBehavior[i];
			tempLike = 0;// 0 means "dislike".
			if (ratingVector[i] > likeThreshold) {
				tempLike = 1;// 1 means "like".
			} // Of if
			resultTotalcost += costMatrix[tempBehavior][tempLike];
		} // Of for i

		return resultTotalcost;
	}// Of computeTotalCost

	/**
	 *********************************** 
	 * Compute the total cost for a user.
	 * 
	 * @paraUser The index of the given user.
	 *********************************** 
	 */
	public double computeTotalCostForUser(int paraUser, boolean[] paraRecommendationList, boolean[] paraPromotionList) {

		// Step 2. Check them.
		double resultTotalcost = 0;
		int tempBehavior;
		int tempLike;
		for (int i = 0; i < numItems; i++) {
			// No rating for this user-item pair
			if (currentUserActualRatings[i] < 1e-6) {
				continue;
			} // Of if

			if (paraRecommendationList[i]) {
				tempBehavior = RECOMMEND;
			} else if (paraPromotionList[i]) {
				tempBehavior = PROMOTE;
			} else {
				tempBehavior = NON_RECOMMEND;
			} // Of if

			tempLike = 0;// 0 means "dislike".
			if (ratingVector[i] > likeThreshold) {
				tempLike = 1;// 1 means "like".
			} // Of if

			resultTotalcost += costMatrix[tempBehavior][tempLike];
			int tempIndex = tempBehavior * 2 + tempLike;
			recommendationSummaryMatrix[paraUser][tempIndex]++;

			// System.out.print(" + " + costMatrix[tempBehavior][tempLike]);
			// System.out.println();
		} // Of for i

		System.out.println(Arrays.toString(recommendationSummaryMatrix[paraUser]));

		return resultTotalcost;
	}// Of computeTotalCost

	/**
	 *********************************** 
	 * Leave-user-out recommendation.
	 * 
	 * @return The total cost of the current user.
	 *********************************** 
	 */
	public double leaveUserOutRecommend() {
		double resultTotalCost = 0;
		for (int i = 0; i < numUsers; i++) {
			if (i % 100 == 0) {
				System.out.println("Recommending for user #" + i + ":");
			} // Of if
			resultTotalCost += recommendForUser(i);
		} // Of for i

		return resultTotalCost;
	}// Of leaveUserOutRecommend

	/**
	 *********************************** 
	 * Recommend for one user.
	 * 
	 * @param paraUser
	 *            The user index.
	 * @return The total cost of the current user.
	 *********************************** 
	 */
	public double recommendForUser(int paraUser) {
		// Step 0. Initialize
		double resultTotalCost;
		// Compute the actual ratings of the current user.
		Arrays.fill(currentUserActualRatings, 0);
		for (int i = userStartingPoints[paraUser]; i < userStartingPoints[paraUser + 1]; i++) {
			currentUserActualRatings[itemVector[i]] = ratingVector[i];
		} // Of for i

		Arrays.fill(currentUserRecommendationList, false);
		Arrays.fill(currentUserPromotionList, false);

		// Step 1. Popularity-based recommendation.
		popBasedRecommend(paraUser);
		resultTotalCost = computeTotalCostForUser(paraUser, currentUserRecommendationList, currentUserPromotionList);
		System.out.print(
				"Only popularity-based recommendtaion. The cost of user " + paraUser + " is: " + resultTotalCost);
		System.out.println();

		// Step 3. M-distance-based recommendation.
		// mDistanceBasedRecommend(paraUser);
		// Step 3. GLMF recommendation.
		GLFactorizationJournal(paraUser);
		resultTotalCost = computeTotalCostForUser(paraUser, currentUserRecommendationList, currentUserPromotionList);
		System.out.print("Finally, the cost of user " + paraUser + " is: " + resultTotalCost);
		System.out.println();

		return resultTotalCost;
		/*
		 * 
		 * // Step 2. M-distance-based recommendation. boolean tempSuccess = true; while
		 * (tempSuccess) { mDistanceBasedRecommend(paraUser); if (3 > 2) { tempSuccess =
		 * false; } // Of if } // Of while
		 */
	}// Of recommendForUser

	/**
	 *********************************** 
	 * Pop-based recommendation. With member variables currentUserRecommendationList
	 * and currentUserPromotionList, there is no need to return.
	 * 
	 * @param paraUser
	 *            The given user.
	 *********************************** 
	 */
	public void popBasedRecommend(int paraUser) {
		// Step 1. Initialize the total/average score of each item.
		int[] tempItemPopArray = new int[numItems];
		for (int i = 0; i < numItems; i++) {
			tempItemPopArray[i] = itemPopArray[i];
		} // Of for i

		// Remove those for the current user since the data is unknown.
		for (int i = userStartingPoints[paraUser]; i < userStartingPoints[paraUser + 1]; i++) {
			tempItemPopArray[itemVector[i]]--;
		} // Of for i

		double tempMaturity = 0;

		// Step 2. Compute popular items.
		int[] tempPopItems = new int[numItems];
		int tempNumPopItems = 0;
		System.out.print("Popular items: ");
		for (int i = 0; i < numItems; i++) {
			if (tempItemPopArray[i] >= popThreshold * maxItemPop) {
				tempPopItems[tempNumPopItems] = i;
				System.out.print(", " + i);
				tempNumPopItems++;
			} // Of for i
		} // Of for i

		// Step 3. Compute semi-popular items.
		int[] tempSemiPopItems = new int[numItems];
		int tempNumSemiPopItems = 0;
		System.out.print("\r\nSemi-popular items: ");
		for (int i = 0; i < numItems; i++) {
			if ((tempItemPopArray[i] < popThreshold * maxItemPop)
					&& (tempItemPopArray[i] >= semiPopThreshold * maxItemPop)) {
				tempSemiPopItems[tempNumSemiPopItems] = i;
				System.out.print("; " + i);
				tempNumSemiPopItems++;
			} // Of for i
		} // Of for i

		// Step 4. Pop-based recommendation.
		int tempNumRecommend = (int) (recommendRatio * recommendationLength);
		int tempNumPromote = recommendationLength - tempNumRecommend;
		boolean[] tempProcessedArray = new boolean[numItems];
		Arrays.fill(tempProcessedArray, false);

		while (tempMaturity < maturityThreshold) {
			if (tempNumRecommend > tempNumPopItems) {
				System.out.println("No enough to recommend.");
				break;
			} // Of if

			// Step 4.1 Randomly select 3 to recommend
			int[] tempRecommendationArray = randomSelectFromArray(tempPopItems, tempNumPopItems, tempNumRecommend);
			for (int i = 0; i < tempRecommendationArray.length; i++) {
				tempProcessedArray[tempRecommendationArray[i]] = true;
				currentUserRecommendationList[tempRecommendationArray[i]] = true;
			} // Of for i
			tempNumPopItems -= tempNumRecommend;
			tempPopItems = eliminateProcessed(tempPopItems, tempProcessedArray);
			System.out.println("\r\nRecommending " + Arrays.toString(tempRecommendationArray));

			// Step 4.2 Promote
			// Step 4.2.1 Compute items with popularity between the pop and
			// promotion thresholds.

			if (tempNumPromote > tempNumSemiPopItems) {
				System.out.println("No enough to promote.");
				break;
			} // Of if

			// Step 4.2.2 Randomly select 7 to promote
			int[] tempPromotionArray = randomSelectFromArray(tempSemiPopItems, tempNumSemiPopItems, tempNumPromote);
			System.out.println("\r\nPromoting " + Arrays.toString(tempPromotionArray));
			tempNumSemiPopItems -= tempNumPromote;
			for (int i = 0; i < tempPromotionArray.length; i++) {
				tempProcessedArray[tempPromotionArray[i]] = true;
				currentUserPromotionList[tempPromotionArray[i]] = true;
			} // Of for i
			tempSemiPopItems = eliminateProcessed(tempSemiPopItems, tempProcessedArray);

			// Step 3 Update the maturity
			tempMaturity += computeUserMaturityIncrement(paraUser, tempRecommendationArray, tempPromotionArray);
		} // Of while

		if (tempMaturity >= maturityThreshold) {
			System.out.println("Matured.");
		} // Of if
	}// Of popBasedRecommend

	/**
	 *********************************** 
	 * Randomly select some elements from the given array.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraValidLength
	 *            Valid length of the array.
	 * @param paraNumSelection
	 *            The number of selected elements.
	 *********************************** 
	 */
	public static int[] randomSelectFromArray(int[] paraArray, int paraValidLength, int paraNumSelection) {
		int[] tempArray = new int[paraValidLength];
		for (int i = 0; i < tempArray.length; i++) {
			tempArray[i] = i;
		} // Of for i
		Random tempRandom = new Random();
		int tempFirstIndex;
		int tempSecondIndex;
		int tempValue;
		for (int i = 0; i < 1000; i++) {
			tempFirstIndex = tempRandom.nextInt(paraValidLength);
			tempSecondIndex = tempRandom.nextInt(paraValidLength);
			tempValue = tempArray[tempFirstIndex];
			tempArray[tempFirstIndex] = tempArray[tempSecondIndex];
			tempArray[tempSecondIndex] = tempValue;
		} // Of for i

		int[] resultArray = new int[paraNumSelection];
		for (int i = 0; i < resultArray.length; i++) {
			resultArray[i] = paraArray[tempArray[i]];
		} // Of for i

		return resultArray;
	}// Of randomSelectFromArray

	/**
	 *********************************** 
	 * Eliminate elements that are already processed.
	 * 
	 * @param paraArray
	 *            The given array.
	 * @param paraProcessed
	 *            Indicate which elements have been processed.
	 * @return The new array.
	 *********************************** 
	 */
	public static int[] eliminateProcessed(int[] paraArray, boolean[] paraProcessed) {
		int[] tempArray = new int[paraArray.length];
		int tempLength = 0;

		// Step 1. Copy unprocessed elements.
		for (int i = 0; i < paraArray.length; i++) {
			if (!paraProcessed[paraArray[i]]) {
				tempArray[tempLength] = paraArray[i];
				tempLength++;
			} // Of if
		} // Of if

		// Step 2. Compress.
		int[] resultArray = new int[tempLength];
		for (int i = 0; i < tempLength; i++) {
			resultArray[i] = tempArray[i];
		} // Of for i
		return resultArray;
	}// Of eliminateProcessed

	/**
	 *********************************** 
	 * Compute the user maturity increment. For each recommendation to existing
	 * rating, the maturity increases by 5. For each recommendation to non-existing
	 * rating, the maturity increases by 2.5. For each promotion to existing rating,
	 * the maturity increases by 4. For each promotion to non-existing rating, the
	 * maturity increases by 2.
	 * 
	 * @param paraUser
	 *            The given user.
	 * @param paraRecommendationArray
	 *            The recommendation array.
	 * @param paraPromotionArray
	 *            The promotion array.
	 *********************************** 
	 */
	public double computeUserMaturityIncrement(int paraUser, int[] paraRecommendationArray, int[] paraPromotionArray) {
		double resultValue = 0;

		for (int i = 0; i < paraRecommendationArray.length; i++) {
			if (currentUserActualRatings[paraRecommendationArray[i]] > 1e-6) {
				resultValue += maturityValueArray[0];
			} else {
				resultValue += maturityValueArray[1];
			} // Of if
		} // Of for i

		for (int i = 0; i < paraPromotionArray.length; i++) {
			if (currentUserActualRatings[paraPromotionArray[i]] > 1e-6) {
				resultValue += maturityValueArray[2];
			} else {
				resultValue += maturityValueArray[3];
			} // Of if
		} // Of for i

		return resultValue;
	}// Of computeUserMaturityIncrement

	/**
	 *********************************** 
	 * Compute the average ratings of items. The given user should not be
	 * considered.
	 * 
	 * @param paraUser
	 *            The given user.
	 *********************************** 
	 *            public double[] computeItemAverageRatings(int paraUser) { double[]
	 *            resultArray = new double[numItems]; for (int i = 0; i <
	 *            resultArray.length; i++) { if (currentUserActualRatings[i] > 1e-6)
	 *            { resultArray[i] = (itemRatingSumArray[i] -
	 *            currentUserActualRatings[i]) / (itemPopArray[i] - 1); } else {
	 *            resultArray[i] = itemRatingSumArray[i] / itemPopArray[i]; }// Of
	 *            if }// Of for i
	 * 
	 *            return resultArray; }// Of computeItemAverageRatings
	 */
	/**
	 * ************************ 
	 * GL Matrix Factorization recommendation.
	 * 
	 * @param paraUser
	 *            The given user 
	 * ************************
	 */

	public void GLFactorizationJournal(int paraUser) {
				// Step 1. Read data and GL transformation..
		Triple[] tempTrainingData = new Triple[numRatings] ;
		for(int i = 0; i < numRatings; i++) {	
				tempTrainingData[i].user = userVector[i];
				tempTrainingData[i].item = itemVector[i];
				if(tempTrainingData[i].user == paraUser) {
					//The current user: if this item has been recommended/promoted to paraUser.
						if (currentUserRecommendationList[tempTrainingData[i].item] ||
								currentUserPromotionList[tempTrainingData[i].item]) {
							tempTrainingData[i].rating = glTransform(ratingVector[i],1, ratingLowerBound,
									ratingUpperBound);
						}//Of if
					
				}else {
					tempTrainingData[i].rating = glTransform(ratingVector[i],1, ratingLowerBound,
							ratingUpperBound);
				}//Of if	
		}//Of for i 
		// Step 2. Matrix Factorization
		
		// Step 3. GL Inverse transformation
		
	}// Of GLFactorizationJournal
	
	/**
	 ************************ 
	 * Transfer a rating value.
	 * 
	 * @param paraValue
	 *            The given value.
	 ************************ 
	 */
	public static double glTransform(double paraValue, double paraV,
			double paraLowerBound, double paraUpperBound) {
		if (paraValue < paraLowerBound + BOUNDARY_VARIATION) {
			paraValue = paraLowerBound + BOUNDARY_VARIATION;
		}// Of if

		if (paraValue > paraUpperBound - BOUNDARY_VARIATION) {
			paraValue = paraUpperBound - BOUNDARY_VARIATION;
		}// Of if

		double resultValue = -Math.log(Math.pow(
				(paraUpperBound - paraLowerBound)
						/ (paraValue - paraLowerBound), paraV) - 1);

		return resultValue;
	}// of glTransform
	
	// /**
	// ***********************************
	// * M-distance-based recommendation.
	// *
	// * @param paraUser
	// * The given user.
	// ***********************************
	// */
	// public void mDistanceBasedRecommend(int paraUser) {
	// //boolean[] tempProcessedArray = new boolean[numItems];
	// //Arrays.fill(tempProcessedArray, false);
	// // Step 1. Initialize the average score of current.
	// //double[] tempItemAverageScoreArray = computeItemAverageRatings(paraUser);
	//
	// //Step 2. Recommend a few rounds.
	// int[] tempRecommendationCandidates = new int[numItems];
	// int tempNumRecommendationCandidates = 0;
	//
	// int[] tempPromotionCandidates = new int[numItems];
	// int tempNumPromotionCandidates = 0;
	// double tempPredict;
	// int tempRound = 0;
	// while (true) {
	// tempRound ++;
	// System.out.println("\r\nRound " + tempRound);
	// tempNumRecommendationCandidates = 0;
	// tempNumPromotionCandidates = 0;
	//
	// //Step 2.1 Predict ratings.
	// for (int i = 0; i < numItems; i++) {
	// //Step 2.1.1 This item has been recommended/promoted to paraUser.
	// if (currentUserRecommendationList[i] || currentUserPromotionList[i]) {
	// continue;
	// }//Of if
	//
	// //Step 2.1.2 Recompute the item's average rating without considering this
	// user.
	// double tempItemAverageRating = 0;
	// if (currentUserActualRatings[i] > 1e-6) {
	// tempItemAverageRating = (itemRatingSumArray[i] - currentUserActualRatings[i])
	// / (itemPopArray[i] - 1);
	// } else {
	// tempItemAverageRating = itemAverageRatingArray[i];
	// }//Of if
	//
	// // Step 2.1.2 Compute neighbors.
	// int[] tempNeighbors = new int[numItems];
	// int tempNumNeighbors = 0;
	// //The score sum of neighbors.
	// double tempScoreSum = 0;
	// for (int j = 0; j < numItems; j++) {
	// //The user does not know the rating of the item yet.
	// if (!currentUserRecommendationList[j] && !currentUserPromotionList[j]) {
	// continue;
	// }//Of if
	//
	// //This item is not rated by the user.
	// if (currentUserActualRatings[j] < 1e-6) {
	// continue;
	// }//Of if
	//
	// //Not a neighbor
	// if ((tempItemAverageRating < itemAverageRatingArray[j] - neighborhoodRadius)
	// || (tempItemAverageRating > itemAverageRatingArray[j] + neighborhoodRadius))
	// {
	// continue;
	// }//Of if
	//
	// //Now record it.
	// tempNeighbors[tempNumNeighbors] = j;
	// tempScoreSum += currentUserActualRatings[j];
	// //System.out.println("Adding " + currentUserActualRatings[j]);
	// tempNumNeighbors ++;
	// }//Of for j
	//
	// // Step 2.1.3 The average score of neighbors as prediction.
	// if (tempNumNeighbors == 0) {
	// tempPredict = -1;
	// } else {
	// tempPredict = tempScoreSum / tempNumNeighbors;
	// }//Of if
	// //System.out.println("tempPredict = " + tempPredict);
	//
	// if (tempPredict >= recommendScoreThreshold) {
	// tempRecommendationCandidates[tempNumRecommendationCandidates] = i;
	// tempNumRecommendationCandidates ++;
	// } else if (tempPredict >= promoteScoreThreshold) {
	// tempPromotionCandidates[tempNumPromotionCandidates] = i;
	// tempNumPromotionCandidates ++;
	// }//Of if
	// }//Of for i
	//
	// int tempNumRecommend = (int) (recommendRatio * recommendationLength);
	// int tempNumPromote = recommendationLength - tempNumRecommend;
	//
	// //System.out.println("tempRecommendationCandidates: " +
	// Arrays.toString(tempRecommendationCandidates));
	// System.out.println("tempNumRecommendationCandidates: " +
	// tempNumRecommendationCandidates);
	// //System.out.println("tempPromotionCandidates: " +
	// Arrays.toString(tempPromotionCandidates));
	// System.out.println("tempNumPromotionCandidates: " +
	// tempNumPromotionCandidates);
	// System.out.println("NumRecommend: " + tempNumRecommend);
	// //No enough to recommend/promote.
	// if ((tempNumRecommendationCandidates < tempNumRecommend) ||
	// (tempNumPromotionCandidates < tempNumPromote)) {
	// System.out.println("No enough items to recommend/promote with M-distance.");
	// break;
	// }//Of if
	//
	// // Step 2.2 Randomly select 3 to recommend
	// int[] tempRecommendationArray =
	// randomSelectFromArray(tempRecommendationCandidates,
	// tempNumRecommendationCandidates, tempNumRecommend);
	// System.out.print("Recommending by M-distance: ");
	// for (int i = 0; i < tempRecommendationArray.length; i++) {
	// currentUserRecommendationList[tempRecommendationArray[i]] = true;
	// System.out.print(", " + tempRecommendationArray[i]);
	// }// Of for i
	//
	// // Step 2.3 Randomly select 7 to promote
	// System.out.print("\r\nPromoting by M-distance: ");
	// int[] tempPromotionArray = randomSelectFromArray(tempPromotionCandidates,
	// tempNumPromotionCandidates, tempNumPromote);
	// for (int i = 0; i < tempPromotionArray.length; i++) {
	// currentUserPromotionList[tempPromotionArray[i]] = true;
	// System.out.print(", " + tempPromotionArray[i]);
	// }// Of for i
	// } // Of while
	//
	// }// Of mDistanceBasedRecommend

	/**
	 *********************************** 
	 * Compute the maturity of the user according to the current recommendation. The
	 * reaction is known from the data. E.g., recommend likes, promote dislikes.
	 * 
	 * @param paraUser
	 *            The index of the user.
	 * @param paraOnceRec
	 *            The recommendation in the last round.
	 * @return The user maturity.
	 *********************************** 
	 */
	public double computeUserMaturity(int paraUser, int[] paraOnceRec) {
		double resultMaturity = 0;
		int tempNumRatedItems = 0;
		int tempRatedFlag = 0;// 0 means that the item is not rated by the user.

		int tempNumPredicDisLike = 0;

		for (int i = 0; i < numItems; i++) {
			if (paraOnceRec[i] == 1 || paraOnceRec[i] == 2) {
				// Promoted or recommended
				for (int j = userStartingPoints[paraUser]; j < userStartingPoints[paraUser + 1]; j++) {
					// Step 1: If the current user has rated the item.
					if (i == itemVector[j]) {
						tempRatedFlag = 1;// 1 means that the item is rated by
											// the user.
						tempNumRatedItems++;
						break;
					} // Of if
				} // Of for j

				// Step 2:
				if (tempRatedFlag == 0) {
					if (itemPopArray[i] > popThreshold * maxItemPop) {
						// The user does not rate a very popular item.
						tempNumPredicDisLike++;
					} // Of if
				} // Of if
			} // Of if
		} // Of for i
		resultMaturity = tempNumRatedItems * maturityForEachItem + tempNumPredicDisLike * maturityForEachItem / 2;

		return resultMaturity;
	}// Of computeUserMaturity

	/**
	 *********************************** 
	 * Show me.
	 *********************************** 
	 */
	public String toString() {
		String resultString = "I am a recommender system.\r\n";
		resultString += "I have " + numUsers + " users, " + numItems + " items, and " + numRatings + " ratings.";
		return resultString;
	}// Of toString

	/**
	 *********************************** 
	 * The main entrance.
	 * 
	 * @throws IOException
	 * @throws NumberFormatException
	 *********************************** 
	 */
	public static void main(String args[]) throws NumberFormatException, IOException {
		ThreeWayInteractiveRecommendation tir = new ThreeWayInteractiveRecommendation(
				"data/movielens100k.data", 943,
				1682, 100000);
		System.out.println(tir);
		// tir.computePopAndSemipopItems(0.8, 0.6);
		tir.setPopThresholds(0.7, 0.5);
		// double tempTotalCost = tir.leaveUserOutRecommend();
		tir.recommendForUser(0);
		double tempTotalCost = tir.computeTotalCost();
		System.out.println("The total cost is: " + tempTotalCost);
	}// Of main

}// Of class ThreeWayInteractiveRecommendation
