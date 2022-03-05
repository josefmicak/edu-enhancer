// See https://aka.ms/new-console-template for more information
using Common.Class;
using DataLayer;

TestData testData = new TestData();
foreach (Test test in testData.Load())
{
    Console.WriteLine(test + "\n");
}

StudentData studentData = new StudentData();
foreach (Student student in studentData.Load())
{
    Console.WriteLine(student + "\n");
}

Console.ReadKey();

/*
TODO:
    - ItemForm.cs
        - LoadItemInfo();
        - LoadDeliveryExecutionInfo();
        - GetAmountOfSubItems();
        - GetResponseIdentifiers();
        - SubitemImages();
        - GetQuestionType();
        - GetChoiceIdentifierValues();
        - FillPossibleAnswerLabel();
        - FillCorrectAnswerLabel();
        - LoadGapIdentifiers();
        - LoadQuestionPoints();
        - SaveQuestionPoints();
        - SaveStudentsPoints();
    + ItemsForm.cs
        + LoadItems(); - itemData.Load()
    - ResultForm.cs
        + LoadStudent(); - studentData.Load()
        - LoadResults();
        + GetTestIdentifiers(); - testData.GetTestNumberIdentifier()
        + DoTestPointsExist(); - test.PointsDetermined
    + StudentForm.cs
        + LoadTestTakers(); - studentData.Load()
    - TestForm.cs
        - LoadResultInfo();
    + TestsForm.cs
        + LoadTests(); - testData.Load()
-------------------------------------------------
    - ItemLoader.cs
    - Item.cs
 */