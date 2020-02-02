package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.Arrays;
import java.util.Date;
import java.util.Properties;

import javax.swing.JComboBox;

import algorithm.*;
import common.*;
import gui.guicommon.*;
import gui.guidialog.common.HelpDialog;
import gui.others.*;

/**
 * The GUI of TCRAlone. <br>
 * Project: Three-way conversational recommendation.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/TCRAlone.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 * @date Created: January 28, 2020. <br>
 *       Last modified: February 1, 2020.
 * @version 1.0
 */

public class TcrGUI implements ActionListener, ItemListener, TextListener {
	/**
	 * The properties for setting.
	 */
	private Properties settings = new Properties();

	/**
	 * Select the arff file.
	 */
	private FilenameField filenameField;

	/**
	 * Eliminate inconsistent instances.
	 */
	private Checkbox compressedFormatCheckbox;

	/**
	 * Number of users.
	 */
	private IntegerField numUsersField;

	/**
	 * Number of items.
	 */
	private IntegerField numItemsField;

	/**
	 * Number of ratings.
	 */
	private IntegerField numRatingsField;

	/**
	 * Cost matrix.
	 */
	double[][] costMatrix = { { 0, 40 }, { 20, 10 }, { 50, 0 } };

	/**
	 * For cost matrix.
	 */
	private DoubleField[][] costMatrixFields;

	/**
	 * Rating bounds.
	 */
	int[] ratingBounds = { -10, 10 };

	/**
	 * For rating bounds.
	 */
	private IntegerField[] ratingBoundFields;

	/**
	 * Popularity threshold.
	 */
	private double[] popularityThresholds = { 0.3, 0.5 };

	/**
	 * For popularity threshold.
	 */
	private DoubleField[] popularityThresholdFields;

	/**
	 * For recommendation list.
	 */
	private IntegerField recommendationLengthField = new IntegerField("10");

	/**
	 * For recommendation list.
	 */
	private DoubleField recommendationRatioField = new DoubleField("0.3");

	/**
	 * Like threshold.
	 */
	private DoubleField likeThresholdField = new DoubleField("1.5");

	/**
	 * Maturity threshold.
	 */
	private DoubleField maturityThresholdField = new DoubleField("1000");

	/**
	 * Prediction threshold.
	 */
	private double[] favoriteThresholds = { -2.0, 0.5 };

	/**
	 * For rating prediction threshold.
	 */
	private DoubleField[] favoriteThresholdFields;

	/**
	 * For rank.
	 */
	private IntegerField rankField;

	/**
	 * For matrix factorization: alpha.
	 */
	private DoubleField alphaField;

	/**
	 * For matrix factorization: lambda.
	 */
	private DoubleField lambdaField;

	/**
	 * For matrix factorization: pretraining rounds.
	 */
	private IntegerField pretrainRoundsField;

	/**
	 * For matrix factorization: incremental training rounds.
	 */
	private IntegerField incrementalTrainRoundsField;

	/**
	 * Matrix factorization algorithm
	 */
	private JComboBox<String> mfAlgorithmJComboBox;

	/**
	 * Checkbox for variable tracking.
	 */
	private Checkbox variableTrackingCheckbox;

	/**
	 * Checkbox for variable tracking.
	 */
	private Checkbox processTrackingCheckbox;

	/**
	 * Result output to file checkbox.
	 */
	private Checkbox fileOutputCheckbox;

	/**
	 * The message area.
	 */
	private TextArea messageTextArea;

	/**
	 * How many times to repeat.
	 */
	private IntegerField repeatTimesField;

	/**
	 *************************** 
	 * The only constructor.
	 *************************** 
	 */
	public TcrGUI() {
		// A simple frame to contain dialogs.
		Frame mainFrame = new Frame();
		mainFrame.setTitle(
				"Three-way conversational recommendation. minfan@swpu.edu.cn, minfanphd@163.com");

		// The top part: select data file.
		Panel sourceFilePanel = new Panel();
		sourceFilePanel.setLayout(new GridLayout(2, 6));
		filenameField = new FilenameField(30);
		filenameField.setText("data/jester-data-1/jester-data-1.txt");
		filenameField.addTextListener(this);

		sourceFilePanel.add(new Label("The .arff file:"));
		sourceFilePanel.add(filenameField);

		Button browseButton = new Button(" Browse ");
		browseButton.setSize(20, 10);
		Panel browsePanel = new Panel();
		browsePanel.add(browseButton);
		browseButton.addActionListener(filenameField);
		sourceFilePanel.add(browsePanel);

		compressedFormatCheckbox = new Checkbox(" Data in compressed format", false);
		sourceFilePanel.add(compressedFormatCheckbox);

		sourceFilePanel.add(new Label(""));
		sourceFilePanel.add(new Label(""));
		sourceFilePanel.add(new Label("Users:"));
		numUsersField = new IntegerField("24983");
		sourceFilePanel.add(numUsersField);

		sourceFilePanel.add(new Label("Items:"));
		numItemsField = new IntegerField("101");
		sourceFilePanel.add(numItemsField);

		sourceFilePanel.add(new Label("Ratings:"));
		numRatingsField = new IntegerField("1810455");
		sourceFilePanel.add(numRatingsField);

		// For cost matrix.
		Panel costMatrixPanel = new Panel();
		costMatrixPanel.setLayout(new GridLayout(2, 6));
		costMatrixFields = new DoubleField[3][2];
		String[][] tempCostLabels = { { "NN:", "NP:" }, { "BN:", "BP:" }, { "PN:", "PP:" } };
		for (int i = 0; i < costMatrixFields.length; i++) {
			for (int j = 0; j < costMatrixFields[0].length; j++) {
				costMatrixFields[i][j] = new DoubleField("" + costMatrix[i][j]);
				costMatrixPanel.add(new Label(tempCostLabels[i][j]));
				costMatrixPanel.add(costMatrixFields[i][j]);
			} // Of for j
		} // Of for i

		Panel ratingBoundsPanel = new Panel();
		ratingBoundsPanel.setLayout(new GridLayout(2, 6));
		ratingBoundFields = new IntegerField[2];
		String[] ratingLabels = { "Rating lower bound: ", "Upper bound: " };
		for (int i = 0; i < 2; i++) {
			ratingBoundsPanel.add(new Label(ratingLabels[i]));
			ratingBoundFields[i] = new IntegerField("" + ratingBounds[i]);
			ratingBoundsPanel.add(ratingBoundFields[i]);
		} // Of for i
		ratingBoundsPanel.add(new Label("Recommendation list length: "));
		ratingBoundsPanel.add(recommendationLengthField);
		ratingBoundsPanel.add(new Label("Recommendation ratio: "));
		ratingBoundsPanel.add(recommendationRatioField);
		ratingBoundsPanel.add(new Label("Like threshold: "));
		ratingBoundsPanel.add(likeThresholdField);
		ratingBoundsPanel.add(new Label("Maturity threshold: "));
		ratingBoundsPanel.add(maturityThresholdField);

		Panel thresholdsPanel = new Panel();
		thresholdsPanel.setLayout(new GridLayout(2, 4));
		popularityThresholdFields = new DoubleField[2];
		String[] popularityThresholdLabels = { "Semi-popular threshold: ", "Popular threshold: " };
		for (int i = 0; i < 2; i++) {
			thresholdsPanel.add(new Label(popularityThresholdLabels[i]));
			popularityThresholdFields[i] = new DoubleField("" + popularityThresholds[i]);
			thresholdsPanel.add(popularityThresholdFields[i]);
		} // Of for i

		favoriteThresholdFields = new DoubleField[2];
		String[] favoriteThresholdLabels = { " Semi-favorite threshold: ",
				" Favorite threshold: " };
		for (int i = 0; i < 2; i++) {
			thresholdsPanel.add(new Label(favoriteThresholdLabels[i]));
			favoriteThresholdFields[i] = new DoubleField("" + favoriteThresholds[i]);
			thresholdsPanel.add(favoriteThresholdFields[i]);
		} // Of for i

		Panel mfParametersPanel = new Panel();
		mfParametersPanel.setLayout(new GridLayout(2, 6));

		mfParametersPanel.add(new Label("MF algorithm: "));
		String[] algorithms = { "Plain MF", "PQ-MF", "GL-MF" };
		mfAlgorithmJComboBox = new JComboBox<String>(algorithms);
		mfAlgorithmJComboBox.setSelectedIndex(0);
		mfParametersPanel.add(mfAlgorithmJComboBox);

		mfParametersPanel.add(new Label("Pretrain rounds: "));
		pretrainRoundsField = new IntegerField("200");
		mfParametersPanel.add(pretrainRoundsField);

		mfParametersPanel.add(new Label("Incremental train rounds: "));
		incrementalTrainRoundsField = new IntegerField("20");
		mfParametersPanel.add(incrementalTrainRoundsField);

		mfParametersPanel.add(new Label("Rank (k):"));
		rankField = new IntegerField("10");
		mfParametersPanel.add(rankField);

		mfParametersPanel.add(new Label("Learning speed (alpha): "));
		alphaField = new DoubleField("0.0001");
		mfParametersPanel.add(alphaField);

		mfParametersPanel.add(new Label("Convergence control (lambda): "));
		lambdaField = new DoubleField("0.005");
		mfParametersPanel.add(lambdaField);

		processTrackingCheckbox = new Checkbox(" Process tracking ", false);
		variableTrackingCheckbox = new Checkbox(" Variable tracking ", false);
		fileOutputCheckbox = new Checkbox(" Output to file ", false);
		Panel trackingPanel = new Panel();
		trackingPanel.add(processTrackingCheckbox);
		trackingPanel.add(variableTrackingCheckbox);
		trackingPanel.add(fileOutputCheckbox);

		Panel topPanel = new Panel();
		topPanel.setLayout(new GridLayout(6, 1));
		topPanel.add(sourceFilePanel);
		topPanel.add(costMatrixPanel);
		topPanel.add(ratingBoundsPanel);
		topPanel.add(thresholdsPanel);
		topPanel.add(mfParametersPanel);
		topPanel.add(trackingPanel);

		Panel centralPanel = new Panel();
		messageTextArea = new TextArea(20, 80);
		centralPanel.add(messageTextArea);

		// The bottom part: ok and exit
		repeatTimesField = new IntegerField("1");
		Panel repeatTimesPanel = new Panel();
		repeatTimesPanel.add(new Label(" Repeat times: "));
		repeatTimesPanel.add(repeatTimesField);

		Button okButton = new Button(" OK ");
		okButton.addActionListener(this);
		// DialogCloser dialogCloser = new DialogCloser(this);
		Button exitButton = new Button(" Exit ");
		// cancelButton.addActionListener(dialogCloser);
		exitButton.addActionListener(ApplicationShutdown.applicationShutdown);
		Button helpButton = new Button(" Help ");
		helpButton.setSize(20, 10);
		helpButton.addActionListener(
				new HelpDialog("Three-way conversational recommendation", "src/gui/TcrHelp.txt"));
		Panel okPanel = new Panel();
		okPanel.add(okButton);
		okPanel.add(exitButton);
		okPanel.add(helpButton);

		Panel southPanel = new Panel();
		southPanel.setLayout(new GridLayout(2, 1));
		southPanel.add(repeatTimesPanel);
		southPanel.add(okPanel);

		mainFrame.setLayout(new BorderLayout());
		mainFrame.add(BorderLayout.NORTH, topPanel);
		mainFrame.add(BorderLayout.CENTER, centralPanel);
		mainFrame.add(BorderLayout.SOUTH, southPanel);

		mainFrame.setSize(800, 700);
		mainFrame.setLocation(10, 10);
		mainFrame.addWindowListener(ApplicationShutdown.applicationShutdown);
		mainFrame.setBackground(GUICommon.MY_COLOR);
		mainFrame.setVisible(true);
	}// Of the constructor

	/**
	 *************************** 
	 * Read the arff file.
	 *************************** 
	 */
	public void actionPerformed(ActionEvent ae) {
		Common.startTime = new Date().getTime();
		messageTextArea.setText("Processing ... Please wait.\r\n");

		// Parameters to be transferred to respective objects.
		String tempFilename = filenameField.getText().trim();
		boolean tempCompressed = compressedFormatCheckbox.getState();
		int tempNumUsers = numUsersField.getValue();
		int tempNumItems = numItemsField.getValue();
		int tempNumRatings = numRatingsField.getValue();
		for (int i = 0; i < 2; i++) {
			ratingBounds[i] = ratingBoundFields[i].getValue();
			popularityThresholds[i] = popularityThresholdFields[i].getValue();
			favoriteThresholds[i] = favoriteThresholdFields[i].getValue();
		} // Of for i

		int tempRecommendationLength = recommendationLengthField.getValue();
		double tempRecommendationRatio = recommendationRatioField.getValue();
		double tempLikeThreshold = likeThresholdField.getValue();
		double tempMaturityThreshold = maturityThresholdField.getValue();
		int tempRank = rankField.getValue();
		double tempAlpha = alphaField.getValue();
		double tempLambda = lambdaField.getValue();
		int tempPretrainRounds = pretrainRoundsField.getValue();
		int tempIncrementalTrainRounds = incrementalTrainRoundsField.getValue();
		int tempAlgorithm = mfAlgorithmJComboBox.getSelectedIndex();

		int tempRepeatTimes = repeatTimesField.getValue();

		SimpleTools.processTracking = processTrackingCheckbox.getState();
		SimpleTools.variableTracking = variableTrackingCheckbox.getState();
		SimpleTools.fileOutput = fileOutputCheckbox.getState();

		// String resultMessage = "";
		double[] tempCostArray = new double[tempRepeatTimes];

		String tempParametersInformation = "Dataset information: filename: " + tempFilename
				+ "\r\n  " + tempNumUsers + " users, " + tempNumItems + " items, " + tempNumRatings
				+ " ratings\r\n  " + "ratings bounds = " + Arrays.toString(ratingBounds) + ", "
				+ "rank = " + tempRank + ", alpha = " + tempAlpha + ", lambda = " + tempLambda
				+ "\r\n  algorithm = " + tempAlgorithm + ", pretrain rounds = "
				+ tempPretrainRounds;
		messageTextArea.append(tempParametersInformation);

		// Read the data here.
		TCR tempTcr = new TCR(tempFilename, tempNumUsers, tempNumItems, tempNumRatings,
				ratingBounds[0], ratingBounds[1], tempLikeThreshold, tempCompressed);
		tempTcr.setCostMatrix(getCostMatrix());

		tempTcr.stage1Recommender.setMaturityThreshold(tempMaturityThreshold);
		tempTcr.stage1Recommender.setRecommendationLengthRatio(tempRecommendationLength,
				tempRecommendationRatio);
		tempTcr.stage1Recommender.setPopularityThresholds(popularityThresholds);

		tempTcr.stage2Recommender.setParameters(tempRank, tempAlpha, tempLambda, tempAlgorithm,
				tempPretrainRounds);
		tempTcr.stage2Recommender.setFavoriteThresholds(favoriteThresholds);
		tempTcr.stage1Recommender.setRecommendationLengthRatio(tempRecommendationLength,
				tempRecommendationRatio);
		tempTcr.stage2Recommender.setIncrementalTrainRounds(tempIncrementalTrainRounds);

		System.out.println("Before pretrain");
		tempTcr.stage2Recommender.pretrain();
		System.out.println("After pretrain");

		double tempMinCost = Double.MAX_VALUE;
		double tempMaxCost = 0;
		double tempCostSum = 0;

		for (int i = 0; i < tempRepeatTimes; i++) {
			tempTcr.reset();
			// tir.computePopAndSemipopItems(0.8, 0.6);
			// double tempTotalCost = tir.leaveUserOutRecommend();

			tempTcr.leaveUserOutRecommend();
			tempCostArray[i] = tempTcr.computeTotalCost();
			tempCostSum += tempCostArray[i];

			messageTextArea.append("\r\n" + i + ": cost = " + tempCostArray[i] + "\r\n"
					+ Arrays.deepToString(tempTcr.getRecommendationStatistics()));

			if (tempMinCost > tempCostArray[i]) {
				tempMinCost = tempCostArray[i];
			} // Of if

			if (tempMaxCost < tempCostArray[i]) {
				tempMaxCost = tempCostArray[i];
			} // Of if
		} // Of for i

		double tempAverageCost = tempCostSum / tempRepeatTimes;
		double tempStandardDeviation = 0;
		double tempDifference = 0;
		for (int i = 0; i < tempRepeatTimes; i++) {
			tempDifference = tempCostArray[i] - tempAverageCost;
			tempStandardDeviation += tempDifference * tempDifference;
		} // Of for i
		tempStandardDeviation /= tempRepeatTimes;
		tempStandardDeviation = Math.sqrt(tempStandardDeviation);

		messageTextArea.append("\r\nSummary:\r\n");
		messageTextArea.append("; Min: " + tempMinCost);
		messageTextArea.append("; Max: " + tempMaxCost + "\r\n");
		messageTextArea.append("; " + tempAverageCost + " +- " + tempStandardDeviation + "\r\n");

		Common.endTime = new Date().getTime();
		long tempTimeUsed = Common.endTime - Common.startTime;
		messageTextArea.append("Runtime: " + tempTimeUsed + "\r\n");

		messageTextArea.append("\r\nEnd.");
	}// Of actionPerformed

	/**
	 *************************** 
	 * When the checkbox is selected or deselected.
	 *************************** 
	 */
	public void itemStateChanged(ItemEvent paraEvent) {
	}// Of itemStateChanged

	/**
	 *************************** 
	 * Set the cost matrix.
	 *************************** 
	 */
	public void setCostMatrix(double[][] paraCostMatrix) {
		costMatrix = paraCostMatrix;
		for (int i = 0; i < costMatrixFields.length; i++) {
			for (int j = 0; j < costMatrixFields[0].length; j++) {
				costMatrixFields[i][j].setText("" + costMatrix[i][j]);
			} // Of for j
		} // Of for i
	}// Of setCostMatrix

	/**
	 *************************** 
	 * Get the cost matrix.
	 *************************** 
	 */
	public double[][] getCostMatrix() {
		for (int i = 0; i < costMatrixFields.length; i++) {
			for (int j = 0; j < costMatrixFields[0].length; j++) {
				costMatrix[i][j] = Double.parseDouble(costMatrixFields[i][j].getText());
			} // Of for j
		} // Of for i

		return costMatrix;
	}// Of getCostMatrix

	/**
	 *************************** 
	 * Read properties to settings.
	 *************************** 
	 */
	public void textValueChanged(TextEvent paraEvent) {
		String tempPropertyFilename = "";
		String tempFilename = filenameField.getText().trim();
		if (tempFilename.indexOf("jester") > 0) {
			tempPropertyFilename = "src/properties/jester.properties";
		} else if (tempFilename.toLowerCase().indexOf("movielens943u1682m") > 0) {
			tempPropertyFilename = "src/properties/MovieLens943u1682m.properties";
		} else if (tempFilename.toLowerCase().indexOf("movielens706u8570m") > 0) {
			tempPropertyFilename = "src/properties/MovieLens706u8570m.properties";
		} else {
			System.out.println("Unknown dataset.");
			return;
		} // Of if

		try {
			InputStream tempInputStream = new BufferedInputStream(
					new FileInputStream(tempPropertyFilename));
			settings.load(tempInputStream);

			compressedFormatCheckbox
					.setState(Boolean.parseBoolean(settings.getProperty("compressed")));

			numUsersField.setText(settings.getProperty("numUsers"));
			numItemsField.setText(settings.getProperty("numItems"));
			numRatingsField.setText(settings.getProperty("numRatings"));

			costMatrixFields[0][0].setText(settings.getProperty("NN"));
			costMatrixFields[0][1].setText(settings.getProperty("NP"));
			costMatrixFields[1][0].setText(settings.getProperty("BN"));
			costMatrixFields[1][1].setText(settings.getProperty("BP"));
			costMatrixFields[2][0].setText(settings.getProperty("PN"));
			costMatrixFields[2][1].setText(settings.getProperty("PP"));

			ratingBoundFields[0].setText(settings.getProperty("ratingLowerBound"));
			ratingBoundFields[1].setText(settings.getProperty("ratingUpperBound"));

			recommendationLengthField.setText(settings.getProperty("recommendationLength"));
			recommendationRatioField.setText(settings.getProperty("recommendationRatio"));

			likeThresholdField.setText(settings.getProperty("likeThreshold"));

			maturityThresholdField.setText(settings.getProperty("maturityThreshold"));

			popularityThresholdFields[0].setText(settings.getProperty("semiPopularThreshold"));
			popularityThresholdFields[1].setText(settings.getProperty("popularThreshold"));

			favoriteThresholdFields[0].setText(settings.getProperty("semiFavoriteThreshold"));
			favoriteThresholdFields[1].setText(settings.getProperty("favoriteThreshold"));

			mfAlgorithmJComboBox
					.setSelectedIndex(Integer.parseInt(settings.getProperty("mfAlgorithm")));

			pretrainRoundsField.setText(settings.getProperty("pretrainRounds"));
			incrementalTrainRoundsField.setText(settings.getProperty("incrementalTrainRounds"));

			rankField.setText(settings.getProperty("rank"));

			alphaField.setText(settings.getProperty("alpha"));
			lambdaField.setText(settings.getProperty("lambda"));
		} catch (Exception ee) {
			System.out.println("Error occurred while reading properties: " + ee);
		} // Of try
	}// Of textValueChanged

	/**
	 *************************** 
	 * The entrance method.
	 * 
	 * @param args
	 *            The parameters.
	 *************************** 
	 */
	public static void main(String args[]) {
		new TcrGUI();
	}// Of main
}// Of class TcrGUI
