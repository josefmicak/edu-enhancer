// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

TestLoader testLoader = new TestLoader();
foreach (Test test in testLoader.LoadTests())
{
    Console.WriteLine(test + "\n");
}

TestTakerLoader testTakerLoader = new TestTakerLoader();
foreach (TestTaker testTaker in testTakerLoader.LoadTestTakers())
{
    Console.WriteLine(testTaker + "\n");
}

Console.ReadKey();

/*
TODO:
    - ItemForm.cs
        - LoadItemInfo();
        - LoadDeliveryExecutionInfo();
        - StudentsAnswerCorrectness
        - LoadDeliveryExecutionInfoToEdit();
        - GetResultsFilePoints();
        - ResetLoadedItemInfo();
        - GetAmountOfSubItems();
        - GetCorrectChoicePoints();
        - GetResponseIdentifiers();
        - SubitemImages();
        - GetQuestionType();
        - SetQuestionTypeLabel();
        - GetChoiceIdentifierValues();
        - FillPossibleAnswerLabel();
        - FillCorrectAnswerLabel();
        - GetChoiceValue();
        - LoadGapIdentifiers();
        - LoadQuestionPoints();
        - SaveQuestionPoints();
        - ...
    + ItemsForm.cs
        + LoadItems();
    - ResultForm.cs
        + LoadStudent();
        - LoadResults();
        + GetTestIdentifiers();
        + DoTestPointsExist();
    + StudentForm.cs
        + LoadTestTakers();
    - TestForm.cs
        - LoadResultInfo();
    + TestsForm.cs
-------------------------------------------------
    - ItemLoader.cs
    - Item.cs
 */