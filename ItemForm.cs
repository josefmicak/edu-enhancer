using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;
using System.Xml.Linq;

namespace TAO_Enhancer
{
    /*
     *Text - při kliknutí na text zobrazit textbox s textem, ochrana toho že je špatné řádkování
     *ResultForm - filtrování podle studenta?
     *Ošteření vstupů
     *Když máme N/A Points.txt, tak ve výsledcích píše 0/N/A namísto N/A (a také questionpointslabel je 0/0)
     */
    public partial class ItemForm : Form
    {
        string itemNumberIdentifier = "";
        string itemNameIdentifier = "";
        string testNameIdentifier = "";
        string testNumberIdentifier = "";
        string subitemIdentifier = "";
        List<string> subquestionArray = new List<string>();
        List<string> possibleAnswerArray = new List<string>();
        List<string> correctChoiceArray = new List<string>();
        List<string> correctAnswerArray = new List<string>();
        List<(string, string)> choiceIdentifierValueTuple = new List<(string, string)>();
        List<string> responseIdentifierArray = new List<string>();
        List<string> responseValueArray = new List<string>();
        int questionType = 0;
        int amountOfSubitems = 0;
        bool subitemsAdded = false;
        List<(bool, string, string)> includesImage = new List<(bool, string, string)>();
        int subquestionPoints = 0;
        int questionPoints = 0;

        bool isTeacherEditingQuestion = true;
        string deliveryExecutionIdentifier = "";
        string studentIdentifier = "";
        List<double> studentsReceivedPointsArray = new List<double>();
        bool deliveryExecutionFileCreated;
        List<double> importedReceivedPointsArray = new List<double>();
        bool isTeacherReviewingDeliveryResult = false;
        double studentsReceivedPoints = 0;
        bool undecidedPointsInFile = false;
        bool negativePoints = false;
        bool recommendedWrongChoicePoints = true;
        double selectedWrongChoicePoints = 0;
        List<string> gapIdentifiers = new List<string>();

        public ItemForm(string testNameID, string itemNameID, string itemNumberID, string testNumberID, bool requestOrigin, string attemptIdentifier, string studentID, bool attemptFileCreated, bool isTeacherEditingDeliveryResult, bool negativePointsInTest)
        {
            InitializeComponent();
            testNameIdentifier = testNameID;
            itemNameIdentifier = itemNameID;
            itemNumberIdentifier = itemNumberID;
            testNumberIdentifier = testNumberID;
            deliveryExecutionFileCreated = attemptFileCreated;
            negativePoints = negativePointsInTest;
            LoadItemInfo();

            if (!requestOrigin)//student
            {
                deliveryExecutionIdentifier = attemptIdentifier;
                isTeacherEditingQuestion = false;
                studentIdentifier = studentID;
                LoadDeliveryExecutionInfo();
                ModifyQuestionGB.Visible = false;

                if(!isTeacherEditingDeliveryResult)
                {
                    NumberIdentifierLabel.Visible = false;
                    ResponseIdentifierLabel.Visible = false;
                }
            }

            if(isTeacherEditingDeliveryResult)
            {
                isTeacherReviewingDeliveryResult = isTeacherEditingDeliveryResult;
                ModifyStudentsPointsGB.Visible = true;
                LoadDeliveryExecutionInfoToEdit();
            }

            string titleText = "TAO Enhancer - Prohlídka otázky " + itemNameIdentifier;
            if(isTeacherEditingDeliveryResult || !requestOrigin)
            {
                titleText += " - Výsledek";
            }
            this.Text = titleText;
        }

        public void LoadItemInfo()
        {
            NumberIdentifierLabel.Text = "Číselný identifikátor: " + itemNumberIdentifier;
            NameIdentifierLabel.Text = "Jmenný identifikátor: " + itemNameIdentifier;

            bool includesQuestion = false;

            ResetLoadedItemInfo();

            amountOfSubitems = GetAmountOfSubitems();
            AmountOfSubquestionsLabel.Text = "Počet podotázek: " + amountOfSubitems;
            if (amountOfSubitems > 1)
            {
                SubitemCB.Enabled = true;
                SubitemLabel.Text = "Vyberte podotázku ze seznamu:";
            }
            else
            {
                SubitemLabel.Text = "Tato otázka obsahuje pouze jednu podotázku.";
            }
            if (!subitemsAdded)
            {
                GetResponseIdentifiers();
                subitemIdentifier = responseIdentifierArray[0];
                ResponseIdentifierLabel.Text = "Identifikátor podotázky: " + responseIdentifierArray[0];
            }
            if (SubitemCB.Enabled && SubitemCB.SelectedIndex == -1)
            {
                SubitemCB.SelectedIndex = 0;
                subitemsAdded = true;
            }

            questionType = GetQuestionType();
            SetQuestionTypeLabel();
            if (questionType == 0)
            {
                MessageBox.Show("Chyba: nepodporovaný nebo neznámý typ otázky.", "Nepodporovaná otázka", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            GetChoiceIdentifierValues();

            int currentSubitem = -1;
            if (amountOfSubitems > 1)
            {
                currentSubitem = SubitemCB.SelectedIndex;
            }
            else
            {
                currentSubitem = 0;
            }

            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (xmlReader.Name == "assessmentItem")
                    {
                        TitleLabel.Text = "Nadpis: " + xmlReader.GetAttribute("title");
                        LabelLabel.Text = "Označení: " + xmlReader.GetAttribute("label");
                    }
                }

                if (includesImage[currentSubitem].Item1)
                {
                    QuestionLabel.Text = "Otázka: " + includesImage[currentSubitem].Item2;
                    QuestionImage.ImageLocation = ("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier + "\\" + includesImage[currentSubitem].Item3);
                    /*if (xmlReader.Name == "div" && xmlReader.AttributeCount == 0 && xmlReader.NodeType != XmlNodeType.EndElement)//TODO 3: Předělat (?), div je potomkem prompt, viz TODO 2
                    {
                        QuestionLabel.Text = "Otázka: " + xmlReader.ReadElementContentAsString();
                    }
                    if (xmlReader.Name == "img")
                    {
                        QuestionImage.ImageLocation = ("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier + "\\" + xmlReader.GetAttribute("src"));
                    }*/
                }
                else
                {
                    if (questionType == 7 || questionType == 8)
                    {
                        if (xmlReader.Name == "p" && xmlReader.NodeType != XmlNodeType.EndElement)
                        {
                            string inlineChoiceInteractionLine = xmlReader.ReadInnerXml();
                            int firstStartTag = inlineChoiceInteractionLine.IndexOf('<');
                            int lastEndTag = inlineChoiceInteractionLine.LastIndexOf('>');
                            string questionText = inlineChoiceInteractionLine.Substring(0, firstStartTag) + "(DOPLŇTE)" + inlineChoiceInteractionLine.Substring(1 + lastEndTag);
                            QuestionLabel.Text = questionText;
                            includesQuestion = true;
                        }
                    }
                    else
                    {
                        if (amountOfSubitems > 1)
                        {
                            string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                            if (responseIdentifier != null && responseIdentifier != responseIdentifierArray[SubitemCB.SelectedIndex])
                            {
                                xmlReader.Skip();
                            }
                        }
                        if (xmlReader.Name == "prompt")
                        {
                            QuestionLabel.Text = "Otázka: " + xmlReader.ReadElementContentAsString();
                            includesQuestion = true;
                        }
                    }
                }

                if(!includesQuestion && !includesImage[currentSubitem].Item1)
                {
                    QuestionLabel.Text = "Otázka nebyla vyplněna.";
                }
            }

            if(questionType == 9)
            {
                if(amountOfSubitems > 1)
                {
                    QuestionLabel.Text = "Otázka: " + responseValueArray[SubitemCB.SelectedIndex];
                }
                else
                {
                    QuestionLabel.Text = "Otázka: " + responseValueArray[0];
                }     
            }

            FillPossibleAnswerLabel();
            FillCorrectAnswerLabel();

            string subquestions = "";
            if (questionType == 4)
            {
                int subquestionArrayIterator = 0;
                foreach (string answer in subquestionArray)
                {
                    if (subquestionArrayIterator != subquestionArray.Count - 1)
                    {
                        subquestions += answer + ", ";
                    }
                    else
                    {
                        subquestions += answer;
                    }
                    subquestionArrayIterator++;
                }
                QuestionLabel.Text += "\n(" + subquestions + ")";
            }

            if(questionType == 9)
            {
                LoadGapIdentifiers();
            }

            LoadQuestionPoints();
        }

        public void LoadDeliveryExecutionInfo()
        {
            StudentsAnswerGB.Visible = true;
            List<string> studentsAnswers = new List<string>();

            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\results"))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string[] attemptIdentifierSplitByUnderscore = Path.GetFileNameWithoutExtension(file).Split("_");
                    if(attemptIdentifierSplitByUnderscore.Length > 2 && attemptIdentifierSplitByUnderscore[2] == deliveryExecutionIdentifier)
                    {
                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if(xmlReader.Name == "itemResult")
                            {
                                if(xmlReader.GetAttribute("identifier") != itemNameIdentifier)
                                {
                                    xmlReader.Skip();
                                }
                            }

                            if(xmlReader.Name == "responseVariable")
                            {
                                if(xmlReader.GetAttribute("identifier") != subitemIdentifier)
                                {
                                    xmlReader.Skip();
                                }
                            }

                            if(xmlReader.Name == "outcomeVariable")
                            {
                                xmlReader.Skip();
                            }

                            if(xmlReader.Name == "value")
                            {
                                string studentsAnswer = xmlReader.ReadElementContentAsString();
                                if(questionType == 3 || questionType == 4 || questionType == 9)
                                {
                                    string[] studentsAnswerSplitBySpace = studentsAnswer.Split(" ");
                                    if (studentsAnswerSplitBySpace.Length == 2)
                                    {
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[1]);
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[0]);
                                    }
                                    else if (studentsAnswerSplitBySpace.Length == 3)
                                    {
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[2]);
                                        studentsAnswers.Add(studentsAnswerSplitBySpace[1]);
                                    }
                                }
                                else if(questionType == 5 || questionType == 8)
                                {
                                    studentsAnswers.Add(studentsAnswer);
                                }
                                else
                                {
                                    string[] studentsAnswerSplitByApostrophe = studentsAnswer.Split("'");
                                    if(studentsAnswerSplitByApostrophe.Length > 1)
                                    {
                                        studentsAnswers.Add(studentsAnswerSplitByApostrophe[1]);
                                    }
                                    else
                                    {
                                        studentsAnswers.Add(studentsAnswer);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < studentsAnswers.Count; i++)
            {
                if(studentsAnswers[i] == "<>")
                {
                    string answer = studentsAnswers[i];
                    studentsAnswers.Remove(answer);
                }
            }

            string studentsAnswerToLabel = "";
            int answerNumber = 0;
            bool studentAnsweredQuestion = false;
            for (int i = 0; i < studentsAnswers.Count; i++)
            {
                for(int j = 0; j < choiceIdentifierValueTuple.Count; j++)
                {
                    if(studentsAnswers[i] == choiceIdentifierValueTuple[j].Item1)
                    {
                        if (questionType == 3 || questionType == 4)
                        {
                            if (answerNumber % 2 == 1)
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + "\n";
                            }
                            else
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + " -> ";
                            }
                            answerNumber++;
                        }
                        else if(questionType == 9)
                        {
                           /* if (answerNumber % 2 == 1)
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + "\n";
                            }
                            else
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + " -> ";
                            }
                            answerNumber++;*/
                        }
                        else
                        {
                            studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + "\n";
                        }
                    }
                }

                if(studentsAnswers[i] != "")
                {
                    studentAnsweredQuestion = true;
                }
            }

            if(questionType == 9)
            {
                int gapNumber = 0;
                for(int i = 0; i < gapIdentifiers.Count; i++)
                {
                    bool gapAnswered = false;
                    for(int j = 0; j < studentsAnswers.Count; j++)
                    {
                        if(j % 2 == 1)
                        {
                            continue;
                        }
                        if(studentsAnswers[j] == gapIdentifiers[i])
                        {
                            for(int k = 0; k < choiceIdentifierValueTuple.Count; k++)
                            {
                                if(studentsAnswers[j + 1] == choiceIdentifierValueTuple[k].Item1)
                                {
                                    gapAnswered = true;
                                    gapNumber++;
                                    studentsAnswerToLabel += "[" + gapNumber + "] - " + choiceIdentifierValueTuple[k].Item2 + "\n";
                                }
                            }
                        }
                    }
                    if(!gapAnswered)
                    {
                        gapNumber++;
                        studentsAnswerToLabel += "[" + gapNumber + "] - nezodpovězeno\n";
                    }
                }
            }

            for(int i = 0; i < studentsAnswers.Count; i++)
            {
                Debug.WriteLine(i + ". studentsAnswers: " + studentsAnswers[i]);
            }
            for (int i = 0; i < choiceIdentifierValueTuple.Count; i++)
            {
                Debug.WriteLine(i + ". choiceIdentifierValueTuple: item 1 - " + choiceIdentifierValueTuple[i].Item1 + ", item 2 - " + choiceIdentifierValueTuple[i].Item2);
            }

     /*       if(questionType == 9)
            {
                studentsAnswerToLabel = "";
                int addedAnswerNumber = 0;
                for (int i = 0; i < studentsAnswers.Count; i++)
                {
                    if (i % 2 == 0)
                    {
                        continue;
                    }
                    for (int j = 0; j < choiceIdentifierValueTuple.Count; j++)
                    {
                        if(studentsAnswers[i] == choiceIdentifierValueTuple[j].Item1)
                        {
                            addedAnswerNumber++;
                            studentsAnswerToLabel += "[" + addedAnswerNumber + "] - " + choiceIdentifierValueTuple[j].Item2 + "\n";
                            break;
                        }
                        if(j == choiceIdentifierValueTuple.Count - 1)
                        {
                            addedAnswerNumber++;
                            studentsAnswerToLabel += "[" + addedAnswerNumber + "] - Nezodpovězeno\n";
                        }
                    }
                }
            }*/


            if (questionType == 5 || questionType == 8)
            {
                studentsAnswerToLabel = studentsAnswers[0];
            }

            if (!studentAnsweredQuestion)
            {
                studentsAnswerToLabel = "Nevyplněno";
            }

            StudentsAnswerLabel.Text = "Vaše odpověď: \n" + studentsAnswerToLabel;

            studentsReceivedPoints = 0;
            StudentsAnswerCorrectness isAnswerCorrect = StudentsAnswerCorrectness.Correct;

            if(deliveryExecutionFileCreated)
            {
                if(importedReceivedPointsArray.Count == 0)
                {
                    double totalReceivedPoints = 0;
                    string[] resultsFileLines = File.ReadAllLines("C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + "Results.txt");
                    for (int i = 0; i < resultsFileLines.Length; i++)
                    {
                        string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                        if (splitResultsFileLineBySemicolon[0] == itemNameIdentifier)
                        {
                            for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                            {
                                importedReceivedPointsArray.Add(double.Parse(splitResultsFileLineBySemicolon[j]));
                                totalReceivedPoints += double.Parse(splitResultsFileLineBySemicolon[j]);
                            }
                        }
                    }

                    QuestionPointsLabel.Text = "Počet bodů za otázku: " + totalReceivedPoints + "/" + questionPoints.ToString();
                }

                if (amountOfSubitems > 1)
                {
                    studentsReceivedPoints = importedReceivedPointsArray[SubitemCB.SelectedIndex];
                }
                else
                {
                    studentsReceivedPoints = importedReceivedPointsArray[0];
                }
            }
            else
            {
                switch (questionType)
                {
                    case int n when (n == 1 || n == 6 || n == 7):
                        bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctChoiceArray, studentsAnswers);
                        if (areStudentsAnswersCorrect)
                        {
                            studentsReceivedPoints = subquestionPoints;
                        }
                        else
                        {
                            if(studentsAnswers.Count == 0 || (studentsAnswers.Count > 0 && studentsAnswers[0] != ""))
                            {
                                if(recommendedWrongChoicePoints)
                                {
                                    studentsReceivedPoints -= subquestionPoints;
                                }
                                else
                                {
                                    studentsReceivedPoints -= Math.Abs(selectedWrongChoicePoints);
                                }
                            }
                        }
                        break;
                    case 2:
                        int studentsCorrectAnswers = 0;

                        for (int i = 0; i < studentsAnswers.Count; i++)
                        {
                            for (int j = 0; j < correctChoiceArray.Count; j++)
                            {
                                if (studentsAnswers[i] == correctChoiceArray[j])
                                {
                                    studentsCorrectAnswers++;
                                    studentsReceivedPoints += ((double)subquestionPoints / (double)correctChoiceArray.Count);
                                }
                            }
                        }

                        if (recommendedWrongChoicePoints)
                        {
                            studentsReceivedPoints -= Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * ((double)subquestionPoints / (double)correctChoiceArray.Count);
                        }
                        else
                        {
                            studentsReceivedPoints -= Math.Abs(Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * (selectedWrongChoicePoints));
                        }
                        break;
                    case int n when (n == 3 || n == 4 || n == 9):
                        studentsCorrectAnswers = 0;

                        for (int i = 0; i < studentsAnswers.Count; i++)
                        {
                            for (int j = 0; j < correctChoiceArray.Count; j++)
                            {
                                if (i % 2 == 0 && j % 2 == 0)
                                {//TODO: viz TODO 10, zde opět využívám sudosti
                                    if ((studentsAnswers[i] == correctChoiceArray[j] && studentsAnswers[i + 1] == correctChoiceArray[j + 1]) ||
                                        (studentsAnswers[i + 1] == correctChoiceArray[j] && studentsAnswers[i] == correctChoiceArray[j + 1]))
                                    {
                                        studentsCorrectAnswers += 2;
                                        studentsReceivedPoints += ((double)subquestionPoints / (double)correctChoiceArray.Count) * 2;
                                    }
                                }
                            }
                        }

                        if (recommendedWrongChoicePoints)
                        {
                            studentsReceivedPoints -= Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * ((double)subquestionPoints / (double)correctChoiceArray.Count);
                        }
                        else
                        {
                            studentsReceivedPoints -= Math.Abs(Math.Abs((studentsAnswers.Count - studentsCorrectAnswers) / 2) * (selectedWrongChoicePoints));
                        }
                        break;
                    case 8:
                        if(correctAnswerArray[0] == studentsAnswers[0])
                        {
                            studentsReceivedPoints = subquestionPoints;
                        }
                        else
                        {
                            if(studentsAnswers.Count > 0 && studentsAnswers[0] != "")
                            {
                                if (recommendedWrongChoicePoints)
                                {
                                    studentsReceivedPoints -= subquestionPoints;
                                }
                                else
                                {
                                    studentsReceivedPoints -= Math.Abs(selectedWrongChoicePoints);
                                }
                            }
                        }
                        break;
                    /*case 9:
                        for(int i = 0; i < studentsAnswers.Count; i++)
                        {
                            Debug.WriteLine(i + ". studentsAnswers: " + studentsAnswers[i]);
                        }
                        for (int i = 0; i < correctChoiceArray.Count; i++)
                        {
                            Debug.WriteLine(i + ". correctChoiceArray: " + correctChoiceArray[i]);
                        }
                        break;*/
                }
            }

            switch (questionType)
            {
                case int n when (n == 1 || n == 6 || n == 7):
                    bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctChoiceArray, studentsAnswers);
                    if (!areStudentsAnswersCorrect)
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    else
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                    }
                    break;
                case 2:
                    int studentsCorrectAnswers = 0;

                    for (int i = 0; i < studentsAnswers.Count; i++)
                    {
                        for (int j = 0; j < correctChoiceArray.Count; j++)
                        {
                            if (studentsAnswers[i] == correctChoiceArray[j])
                            {
                                studentsCorrectAnswers++;
                            }
                        }
                    }

                    if (studentsCorrectAnswers == 0)
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    else
                    {
                        if (studentsReceivedPoints != subquestionPoints || studentsAnswers.Count != correctChoiceArray.Count)
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.PartiallyCorrect;
                        }
                        else
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                        }
                    }
                    break;
                case int n when (n == 3 || n == 4 || n == 9):
                    studentsCorrectAnswers = 0;

                    for (int i = 0; i < studentsAnswers.Count; i++)
                    {
                        for (int j = 0; j < correctChoiceArray.Count; j++)
                        {
                            if (i % 2 == 0 && j % 2 == 0)
                            {//TODO: viz TODO 10, zde opět využívám sudosti
                                if ((studentsAnswers[i] == correctChoiceArray[j] && studentsAnswers[i + 1] == correctChoiceArray[j + 1]) ||
                                    (studentsAnswers[i + 1] == correctChoiceArray[j] && studentsAnswers[i] == correctChoiceArray[j + 1]))
                                {
                                    studentsCorrectAnswers += 2;
                                }
                            }
                        }
                    }

                    if (studentsCorrectAnswers == 0)
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    else
                    {
                        if (studentsReceivedPoints != subquestionPoints || studentsAnswers.Count != correctChoiceArray.Count)
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.PartiallyCorrect;
                        }
                        else
                        {
                            isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                        }
                    }
                    break;
                case 5:
                    isAnswerCorrect = StudentsAnswerCorrectness.Unknown;
                    break;
                case 8:
                    if (correctAnswerArray[0] == studentsAnswers[0])
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                    }
                    else
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    break;
            }

            switch (isAnswerCorrect)
            {
                case StudentsAnswerCorrectness.Correct:
                    StudentsAnswerCorrectLabel.Text = "Správná odpověď.";
                    break;
                case StudentsAnswerCorrectness.PartiallyCorrect:
                    StudentsAnswerCorrectLabel.Text = "Částečně správná odpověď.";
                    break;
                case StudentsAnswerCorrectness.Incorrect:
                    StudentsAnswerCorrectLabel.Text = "Nesprávná odpověď.";
                    break;
                case StudentsAnswerCorrectness.Unknown:
                    StudentsAnswerCorrectLabel.Text = "Otevřená odpověď, body budou přiděleny manuálně.";
                    break;
            }

            if ((studentsReceivedPoints < 0 && !negativePoints) || (studentsAnswers.Count > 0 && studentsAnswers[0] == ""))
            {
                studentsReceivedPoints = 0;
            }

            if (undecidedPointsInFile)
            {
                StudentsAnswerPointstLabel.Text = "Počet bodů za odpověď: N/A";
            }
            else
            {
                StudentsAnswerPointstLabel.Text = "Počet bodů za odpověď: " + studentsReceivedPoints + "/" + subquestionPoints;
            }
            studentsReceivedPointsArray.Add(Math.Round(studentsReceivedPoints, 2));
        }

        enum StudentsAnswerCorrectness
        {
            Correct,
            Incorrect,
            PartiallyCorrect,
            Unknown
        }

        public void LoadDeliveryExecutionInfoToEdit()
        {
            StudentsPointsTB.Text = studentsReceivedPoints.ToString();
        }

        public List<double> GetResultsFilePoints()
        {
            if(amountOfSubitems > 1)
            {
                for(int i = 1; i < SubitemCB.Items.Count; i++)
                {
                    SubitemCB.SelectedIndex = i;
                }
            }
            return studentsReceivedPointsArray;
        }

        public string GetItemPath()
        {
            return "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier + "\\qti.xml";
        }

        public void ResetLoadedItemInfo()
        {
            subquestionArray.Clear();
            possibleAnswerArray.Clear();
            correctChoiceArray.Clear();
            correctAnswerArray.Clear();
            choiceIdentifierValueTuple.Clear();

            QuestionImage.Image = null;
            CorrectAnswerLabel.Text = "";
            CorrectAnswerLabel.Visible = true;
        }

        public int GetAmountOfSubitems()
        {
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "responseDeclaration" && xmlReader.NodeType != XmlNodeType.EndElement && !subitemsAdded)
                {
                    amountOfSubitems++;
                }
            }
            return amountOfSubitems;
        }

        public double GetCorrectChoicePoints()
        {
            double correctChoicePoints = 0;
            switch (questionType)
            {
                case int n when (n == 1 || n == 5 || n == 6 || n == 7 || n == 8):
                    correctChoicePoints = subquestionPoints;
                    break;
                case 2:
                    correctChoicePoints = (double)subquestionPoints / (double)correctChoiceArray.Count;
                    break;
                case int n when (n == 3 || n == 4 || n == 9):
                    correctChoicePoints = (double)subquestionPoints / (double)(correctChoiceArray.Count / 2);
                    break;
            }
            if(correctChoicePoints == Double.PositiveInfinity || correctChoicePoints == Double.NegativeInfinity)
            {
                string errorMessage = "Chyba: otázka nemá pravděpodobně zadané žádné správné odpovědi.\nIdentifikátory otázky: " + itemNameIdentifier + ", " + itemNumberIdentifier + "\nAplikace bude nyní ukončena.";
                MessageBox.Show(errorMessage, "Neexistující správné odpovědi", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Environment.Exit(0);
            }
            return Math.Round(correctChoicePoints, 2);
        }

        public void GetResponseIdentifiers()
        {
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "responseDeclaration" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    xmlReader.Skip();
                }

                if (xmlReader.GetAttribute("responseIdentifier") != null)
                {
                    string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                    responseIdentifierArray.Add(responseIdentifier);
                    (int amountOfImages, string questionText, _) = SubitemImages(responseIdentifier);

                    if (responseIdentifierArray.Count - 1 > responseValueArray.Count)
                    {
                        responseValueArray.Add("Otázka nebyla vyplněna.");
                    }
                }

                if(responseIdentifierArray.Count > 0)
                {
                    if (includesImage.Count == 0)
                    {
                        string errorMessage = "Chyba: otázka nemá pravděpodobně zadaný žádný text.\nIdentifikátory otázky: " + itemNameIdentifier + ", " + itemNumberIdentifier + "\nAplikace bude nyní ukončena.";
                        MessageBox.Show(errorMessage, "Příliš vysoký počet obrázků", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        Environment.Exit(0);
                    }
                    else
                    {
                        if (includesImage[responseIdentifierArray.Count - 1].Item1)
                        {
                            try
                            {
                                if (xmlReader.Name == "div" && xmlReader.AttributeCount == 0 && xmlReader.NodeType != XmlNodeType.EndElement)
                                {
                                    string responseValue = xmlReader.ReadElementContentAsString();
                                    responseValueArray.Add(responseValue);
                                }
                            }
                            catch
                            {
                                string responseValue = includesImage[responseIdentifierArray.Count - 1].Item2;
                                responseValueArray.Add(responseValue);
                            }
                        }
                        else
                        {
                            if (xmlReader.Name == "prompt")
                            {
                                string responseValue = xmlReader.ReadElementContentAsString();
                                responseValueArray.Add(responseValue);
                            }

                            if(xmlReader.Name == "gapMatchInteraction")
                            {
                                using (var innerReader = xmlReader.ReadSubtree())
                                {
                                    while (innerReader.Read())
                                    {
                                        if(innerReader.Name == "p")
                                        {
                                            string gapText = innerReader.ReadInnerXml();
                                            string questionText = "";
                                            bool addText = true;
                                            int gapCounter = 1;

                                            for (int i = 0; i < gapText.Length; i++)
                                            {
                                                if(gapText[i] == '<')
                                                {
                                                    addText = false;
                                                    questionText += "(DOPLŇTE [" + gapCounter + "])";
                                                    gapCounter++;
                                                }
                                                if (gapText[i] == '>')
                                                {
                                                    addText = true;
                                                    continue;
                                                }
                                                if(addText)
                                                {
                                                    questionText += gapText[i];
                                                }
                                            }
                                            responseValueArray.Add(questionText);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for(int i = 0; i < includesImage.Count; i++)
            {
                if(includesImage[i].Item1 && includesImage[i].Item2 == "")
                {
                    includesImage[i] = (includesImage[i].Item1, responseValueArray[i], includesImage[i].Item3);
                }

                if (includesImage[i].Item1 && includesImage[i].Item2 == "")
                {
                    XmlReader xmlReaderCorrection = XmlReader.Create(GetItemPath());
                    while (xmlReaderCorrection.Read())
                    {
                        if(xmlReaderCorrection.Name == "choiceInteraction")
                        {
                            if(xmlReaderCorrection.GetAttribute("responseIdentifier") != responseIdentifierArray[i])
                            {
                                xmlReaderCorrection.Skip();
                            }
                        }
                        if (xmlReaderCorrection.Name == "prompt")
                        {
                            string promptQuestionText = xmlReaderCorrection.ReadInnerXml();
                            int firstStartTag = promptQuestionText.IndexOf('<');
                            int lastEndTag = promptQuestionText.LastIndexOf('>');
                            string questionText = promptQuestionText.Substring(0, firstStartTag) + promptQuestionText.Substring(1 + lastEndTag);
                            responseValueArray[i] = questionText;
                            includesImage[i] = (includesImage[i].Item1, responseValueArray[i], includesImage[i].Item3);
                            break;
                        }
                    }
                }
            }

            bool subquestionWithoutText = false;
            for(int i = 0; i < responseValueArray.Count; i++)
            {
                if(responseValueArray[i] == "")
                {
                    subquestionWithoutText = true;
                }
            }

            if(subquestionWithoutText)
            {
                MessageBox.Show("Chyba: některá z načtených otázek nemá přidělený text (zadání). Může se jednat o chybu způsobenou chybným umístěním obrázku v rámci zadání otázky.", "Otázka bez textu", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }

            if (SubitemCB.Enabled)
            {
                int i = 1;
                foreach (string responseValue in responseValueArray)
                {
                    SubitemCB.Items.Add(i + ") " + responseValue);
                    i++;
                }
            }
        }

        public (int, string, string) SubitemImages(string responseIdentifier)
        {
            int amountOfImages = 0;
            string questionText = "";
            string imageSource = "";

            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction")
                    {
                        if(xmlReader.GetAttribute("responseIdentifier") != responseIdentifier)
                        {
                            xmlReader.Skip();
                        }
                        else
                        {
                            using (var innerReader = xmlReader.ReadSubtree())
                            {
                                while (innerReader.ReadToFollowing("prompt"))
                                {
                                    using (var innerReaderNext = innerReader.ReadSubtree())
                                    {
                                        while(innerReaderNext.Read())
                                        {
                                            if(innerReaderNext.Name == "img")
                                            {
                                                imageSource = innerReaderNext.GetAttribute("src");
                                                amountOfImages++;
                                            }
                                            if (innerReaderNext.Name == "div")
                                            {
                                             //   questionText = innerReaderNext.ReadString();
                                                using (var innerReaderNextNext = innerReader.ReadSubtree())
                                                {
                                                    while (innerReaderNextNext.Read())
                                                    {
                                                        if (innerReaderNext.Name == "img")
                                                        {
                                                            imageSource = innerReaderNext.GetAttribute("src");
                                                            amountOfImages++;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if(amountOfImages > 1)
            {
                string errorMessage = "Chyba: podotázka může obsahovat nanejvýš jeden obrázek.\nIdentifikátory otázky: " + itemNameIdentifier + ", " + itemNumberIdentifier + "\nPočet obrázků: " + amountOfImages + "\nAplikace bude nyní ukončena.";
                MessageBox.Show(errorMessage, "Příliš vysoký počet obrázků", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Environment.Exit(0);
            }
            else if (amountOfImages == 1)
            {
                includesImage.Add((true, questionText, imageSource));
            }
            else
            {
                includesImage.Add((false, questionText, imageSource));
            }

            return (amountOfImages, questionText, imageSource);
        }

        public int GetQuestionType()
        {
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            bool singleCorrectAnswer = false;//questionType = 6 nebo 7; jediná správná odpověď
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes && xmlReader.Name == "responseDeclaration")
                {
                    if (amountOfSubitems > 1)
                    {
                        string responseIdentifier = xmlReader.GetAttribute("identifier");
                        if (responseIdentifier != null && responseIdentifier != responseIdentifierArray[SubitemCB.SelectedIndex])
                        {
                            xmlReader.Skip();
                        }
                    }

                    if (xmlReader.GetAttribute("cardinality") == "ordered" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        questionType = 1;//Typ otázky = seřazení pojmů
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        questionType = 2;//Typ otázky = více odpovědí (abc); více odpovědí může být správně
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "pair")
                    {
                        questionType = 3;//Typ otázky = spojování párů
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "directedPair")
                    {
                        questionType = 4;//Typ otázky = více otázek (tabulka); více odpovědí může být správně
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "string")
                    {
                        questionType = 5;//Typ otázky = volná odpověď; odpověď není předem daná
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        singleCorrectAnswer = true;
                    }
                }

                if(xmlReader.Name == "gapMatchInteraction")
                {
                    if (amountOfSubitems > 1)
                    {
                        string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                        if (responseIdentifier == responseIdentifierArray[SubitemCB.SelectedIndex])
                        {
                            questionType = 9;
                        }
                    }
                }

                if (singleCorrectAnswer)
                {
                    if (xmlReader.Name == "simpleChoice")
                    {
                        questionType = 6;//Typ otázky = výběr z více možností (abc), jen jedna odpověď je správně
                    }
                }

                if (xmlReader.Name == "textEntryInteraction" && questionType == 5)
                {
                    questionType = 8;//Typ otázky = volná odpověď; odpověď je předem daná
                }

                if (amountOfSubitems > 1)
                {
                    string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                    if (responseIdentifier != null && responseIdentifier != responseIdentifierArray[SubitemCB.SelectedIndex])
                    {
                        xmlReader.Skip();
                    }
                }
            }

            if (singleCorrectAnswer && questionType == 0)
            {
                questionType = 7;//Typ otázky = výběr z více možností (dropdown), jen jedna odpověď je správně
            }
            return questionType;
        }

        public void SetQuestionTypeLabel()
        {
            QuestionTypeLabel.Text = "Typ otázky: ";
            switch (questionType)
            {
                case 0:
                    QuestionTypeLabel.Text = "Neznámý nebo nepodporovaný typ otázky!";
                    break;
                case 1:
                    QuestionTypeLabel.Text += "Seřazení pojmů";
                    break;
                case 2:
                    QuestionTypeLabel.Text += "Výběr z více možností; více možných správných odpovědí";
                    break;
                case 3:
                    QuestionTypeLabel.Text += "Spojování pojmů";
                    break;
                case 4:
                    QuestionTypeLabel.Text += "Více otázek k jednomu pojmu; více možných správných odpovědí";
                    break;
                case 5:
                    QuestionTypeLabel.Text += "Volná odpověď, správná odpověď není automaticky určena";
                    break;
                case 6:
                    QuestionTypeLabel.Text += "Výběr z více možností; jedna správná odpověd";
                    break;
                case 7:
                    QuestionTypeLabel.Text += "Výběr z více možností (doplnění textu); jedna správná odpověd";
                    break;
                case 8:
                    QuestionTypeLabel.Text += "Volná odpověď, správná odpověď je automaticky určena";
                    break;
                case 9:
                    QuestionTypeLabel.Text += "Dosazování pojmů do mezer";
                    break;
            }
        }

        public void GetChoiceIdentifierValues()
        {
            if(questionType == 7)
            {
                XmlReader xmlReaderInlineChoice = XmlReader.Create(GetItemPath());
                while (xmlReaderInlineChoice.Read())
                {
                    if (xmlReaderInlineChoice.NodeType == XmlNodeType.Element)
                    {
                        var name = xmlReaderInlineChoice.Name;
                        if (name == "p")
                        {
                            using (var innerReader = xmlReaderInlineChoice.ReadSubtree())
                            {
                                while (innerReader.ReadToFollowing("inlineChoiceInteraction"))
                                {
                                    using (var innerReaderNext = innerReader.ReadSubtree())
                                    {
                                        while (innerReaderNext.ReadToFollowing("inlineChoice"))
                                        {
                                            string choiceIdentifier = innerReaderNext.GetAttribute("identifier");
                                            string choiceValue = innerReaderNext.ReadString();
                                            choiceIdentifierValueTuple.Add((choiceIdentifier, choiceValue));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                XmlReader xmlReader = XmlReader.Create(GetItemPath());
                while (xmlReader.Read())
                {
                    if (xmlReader.Name == "simpleChoice" || xmlReader.Name == "simpleAssociableChoice" || xmlReader.Name == "gapText")
                    {
                        string choiceIdentifier = xmlReader.GetAttribute("identifier");
                        string choiceValue = xmlReader.ReadElementContentAsString();
                        choiceIdentifierValueTuple.Add((choiceIdentifier, choiceValue));
                    }
                }
            }
        }

        public void FillPossibleAnswerLabel()
        {
            string possibleAnswers = "";
            int simpleMatchSetCounter = 0;

            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (amountOfSubitems > 1)
                {
                    string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                    if (responseIdentifier != null && responseIdentifier != responseIdentifierArray[SubitemCB.SelectedIndex])
                    {
                        xmlReader.Skip();
                    }
                }

                if (questionType == 4)
                {
                    if (xmlReader.Name == "simpleMatchSet")
                    {
                        simpleMatchSetCounter++;
                    }

                    if (xmlReader.Name == "simpleAssociableChoice")
                    {
                        if (simpleMatchSetCounter == 1)
                        {
                            string answerText = xmlReader.ReadElementContentAsString();
                            possibleAnswerArray.Add(answerText);
                        }
                        else if (simpleMatchSetCounter == 3)
                        {
                            string answerText = xmlReader.ReadElementContentAsString();
                            subquestionArray.Add(answerText);
                        }
                    }
                }
                else
                {
                    if (xmlReader.Name == "simpleChoice" || xmlReader.Name == "simpleAssociableChoice" || xmlReader.Name == "gapText")
                    {
                        string answerText = xmlReader.ReadElementContentAsString();
                        possibleAnswerArray.Add(answerText);
                    }
                }
            }

            if (questionType == 4)
            {
                foreach (string answer in possibleAnswerArray)
                {
                    possibleAnswers += answer + "\n";
                }
                PossibleAnswerLabel.Text = "Možné odpovědi:\n" + possibleAnswers;
            }
            else if (questionType == 5)
            {
                PossibleAnswerLabel.Text = "Jedná se o otevřenou otázku, neobsahuje výběr z možností, odpovědi je nutné ověřit manuálně.";
            }
            else if (questionType == 8)
            {
                PossibleAnswerLabel.Text = "Otázka neobsahuje výběr z možností.";
            }
            else
            {
                foreach (string answer in possibleAnswerArray)
                {
                    possibleAnswers += answer + "\n";
                }
                PossibleAnswerLabel.Text = "Možné odpovědi:\n" + possibleAnswers;
            }

            if(questionType == 7)
            {
                XmlReader xmlReaderInlineChoice = XmlReader.Create(GetItemPath());
                while (xmlReaderInlineChoice.Read())
                {
                    if (xmlReaderInlineChoice.NodeType == XmlNodeType.Element)
                    {
                        var name = xmlReaderInlineChoice.Name;
                        if (name == "p")
                        {
                            using (var innerReader = xmlReaderInlineChoice.ReadSubtree())
                            {
                                while (innerReader.ReadToFollowing("inlineChoiceInteraction"))
                                {
                                    using (var innerReaderNext = innerReader.ReadSubtree())
                                    {
                                        while (innerReaderNext.ReadToFollowing("inlineChoice"))
                                        {
                                              possibleAnswerArray.Add(innerReaderNext.ReadString());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                foreach (string answer in possibleAnswerArray)
                {
                    possibleAnswers += answer + "\n";
                }
                PossibleAnswerLabel.Text = "Možné odpovědi:\n" + possibleAnswers;
            }
        }

        public void FillCorrectAnswerLabel()
        {
            string correctAnswer = "";
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (amountOfSubitems > 1 && xmlReader.Name == "responseDeclaration")
                {
                    string responseIdentifier = xmlReader.GetAttribute("identifier");
                    if (responseIdentifier != null && responseIdentifier != responseIdentifierArray[SubitemCB.SelectedIndex])
                    {
                        xmlReader.Skip();
                    }
                }

                if (questionType == 3 || questionType == 4)
                {
                    if (xmlReader.Name == "value")
                    {
                        string value = xmlReader.ReadElementContentAsString();
                        if (value.Length > 1)//TODO 8: viz TODO 2
                        {
                            string[] orderedCorrectChoices = value.Split(' ');
                            correctChoiceArray.Add(orderedCorrectChoices[1]);
                            correctChoiceArray.Add(orderedCorrectChoices[0]);
                        }
                    }
                }
                else if (questionType == 5)
                {
                    CorrectAnswerLabel.Visible = false;
                }
                else if(questionType == 8)
                {
                    if (xmlReader.Name == "value")
                    {
                        string value = xmlReader.ReadElementContentAsString();
                        if (value.Length > 1)//TODO 8: viz TODO 2
                        {
                            correctAnswerArray.Add(value);
                        }
                    }
                }
                else if (questionType == 9)
                {
                    if (xmlReader.Name == "value")
                    {
                        string value = xmlReader.ReadElementContentAsString();
                        if (value.Length > 1)//TODO 8: viz TODO 2
                        {
                            string[] orderedCorrectChoices = value.Split(' ');
                            correctChoiceArray.Add(orderedCorrectChoices[0]);
                            correctChoiceArray.Add(orderedCorrectChoices[1]);
                     //       correctAnswerArray.Add(" ");//placeholder
                        }
                    }

                    if (xmlReader.Name == "gapText")
                    {
                        for(int i = 0; i < correctChoiceArray.Count; i++)
                        {
                            if(i % 2 == 1)
                            {
                                continue;
                            }

                            if (xmlReader.GetAttribute("identifier") == correctChoiceArray[i])
                            {
                                correctAnswerArray.Add(xmlReader.ReadElementContentAsString());
                            }
                        }
                       /* int i = 0;
                        foreach (string answer in correctChoiceArray)
                        {
                         /*   if(i % 2 == 1)
                            {
                                i++;
                                continue;
                            }
                            if (xmlReader.GetAttribute("identifier") == answer)
                            {
                                string answerText = xmlReader.ReadElementContentAsString();
                                correctAnswerArray[i] = answerText;
                            }
                            i++;
                        }*/
                    }
                }
                else
                {
                    if (xmlReader.Name == "value")
                    {
                        string value = xmlReader.ReadElementContentAsString();
                        if (value.Length > 1)//TODO 2: Později předělat kód, ať to zjišťuje jestli jsme v correctResponse (správně) nebo v defaultValue (špatně)
                        {
                            correctChoiceArray.Add(value);
                            correctAnswerArray.Add(" ");//placeholder
                        }
                    }

                    if (xmlReader.Name == "simpleChoice")
                    {
                        int i = 0;
                        foreach (string answer in correctChoiceArray)
                        {
                            if (xmlReader.GetAttribute("identifier") == answer)
                            {
                                string answerText = xmlReader.ReadElementContentAsString();
                                correctAnswerArray[i] = answerText;
                            }
                            i++;
                        }
                    }
                }
            }

            if(questionType == 7)
            {
                correctAnswerArray.Clear();

                XmlReader xmlReaderInlineChoice = XmlReader.Create(GetItemPath());
                while (xmlReaderInlineChoice.Read())
                {
                    if (xmlReaderInlineChoice.NodeType == XmlNodeType.Element)
                    {
                        var name = xmlReaderInlineChoice.Name;
                        if (name == "p")
                        {
                            using (var innerReader = xmlReaderInlineChoice.ReadSubtree())
                            {
                                while (innerReader.ReadToFollowing("inlineChoiceInteraction"))
                                {
                                    using (var innerReaderNext = innerReader.ReadSubtree())
                                    {
                                        while (innerReaderNext.ReadToFollowing("inlineChoice"))
                                        {
                                            for(int i = 0; i < correctChoiceArray.Count; i++)
                                            {
                                                if(innerReaderNext.GetAttribute("identifier") == correctChoiceArray[i])
                                                {
                                                    correctAnswerArray.Add(innerReaderNext.ReadString());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //TODO 6: Zde máme sudý počet pojmů, a máme z nich vytvořit dvojice.
            //Já využívám té sudosti a mám všechny možnosti v poli stringů, možná by bylo vhodnejší použití pole tuplů stringů
            if (questionType == 3 || questionType == 4)
            {
                foreach (string answer in correctChoiceArray)
                {
                    correctAnswerArray.Add(GetChoiceValue(answer));
                }
                int answerNumber = 0;
                foreach (string answer in correctAnswerArray)
                {
                    if (answerNumber % 2 == 1)
                    {
                        correctAnswer += answer + "\n";
                    }
                    else
                    {
                        correctAnswer += answer + " -> ";
                    }
                    answerNumber++;
                }
                CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;
            }
            else if(questionType == 9)
            {
                int answerNumber = 1;
                foreach (string answer in correctAnswerArray)
                {
                    correctAnswer += "[" + answerNumber + "] - " + answer + "\n";
                    answerNumber++;
                }
                CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;
            }
            else
            {
                foreach (string answer in correctAnswerArray)
                {
                    correctAnswer += answer + "\n";
                }
                CorrectAnswerLabel.Text = "Správná odpověď:\n" + correctAnswer;
            }
        }

        public string GetChoiceValue(string choiceIdentifier)
        {
            for (int i = 0; i < choiceIdentifierValueTuple.Count; i++)
            {
                if (choiceIdentifier == choiceIdentifierValueTuple[i].Item1)
                {
                    return choiceIdentifierValueTuple[i].Item2;
                }
            }
            return "Chyba";
        }

        public void LoadGapIdentifiers()
        {
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if(xmlReader.Name == "gap")
                {
                    gapIdentifiers.Add(xmlReader.GetAttribute("identifier"));
                }
            }
        }

        public void LoadQuestionPoints()
        {
            if(amountOfSubitems > 1)
            {
                subitemIdentifier = responseIdentifierArray[SubitemCB.SelectedIndex];
            }

            bool fileExists = false;
            bool itemRecordExists = false;
            undecidedPointsInFile = false;
            questionPoints = 0;
            string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier;
            foreach (var file in Directory.GetFiles(itemParentPath))
            {
                if(Path.GetFileName(file) == "Points.txt")
                {
                    fileExists = true;
                }
            }

            if (!fileExists)
            {
                string itemPointsText = subitemIdentifier + ";N/A;N/A" + Environment.NewLine;
                File.WriteAllText(itemParentPath + "\\Points.txt", itemPointsText);
                SubquestionPointsLabel.Text = "Počet bodů za podotázku: N/A";
                SubquestionPointsTB.Text = "N/A";
            }
            else
            {
                string[] importedFileLines = File.ReadAllLines(itemParentPath + "\\Points.txt");
                for(int i = 0; i < importedFileLines.Length; i++)
                {
                    string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                    if(splitImportedFileLineBySemicolon[0] == subitemIdentifier)
                    {
                        itemRecordExists = true;
                        SubquestionPointsTB.Text = splitImportedFileLineBySemicolon[1];
                        SubquestionPointsLabel.Text = "Počet bodů za podotázku: " + splitImportedFileLineBySemicolon[1];

                        if (splitImportedFileLineBySemicolon[1] != "N/A")
                        {
                            subquestionPoints = int.Parse(splitImportedFileLineBySemicolon[1]);
                        }

                        if (splitImportedFileLineBySemicolon[1] == "N/A")
                        {
                            undecidedPointsInFile = true;
                        }

                        if(splitImportedFileLineBySemicolon.Length > 2 && splitImportedFileLineBySemicolon[2] != "N/A")
                        {
                            RecommendedWrongChoicePointsRB.Enabled = true;
                            SelectedWrongChoicePointsRB.Enabled = true;
                            if(SelectedWrongChoicePointsRB.Checked)
                            {
                                SelectedWrongChoicePointsTB.ReadOnly = false;
                            }
                            if (double.Parse(splitImportedFileLineBySemicolon[2]) == GetCorrectChoicePoints() * (-1))
                            {
                                RecommendedWrongChoicePointsTB.Text = splitImportedFileLineBySemicolon[2];
                                RecommendedWrongChoicePointsRB.Checked = true;
                                recommendedWrongChoicePoints = true;
                            }
                            else
                            {
                                RecommendedWrongChoicePointsTB.Text = (GetCorrectChoicePoints() * (-1)).ToString();
                                SelectedWrongChoicePointsTB.Text = splitImportedFileLineBySemicolon[2];
                                SelectedWrongChoicePointsRB.Checked = true;
                                recommendedWrongChoicePoints = false;
                                selectedWrongChoicePoints = double.Parse(splitImportedFileLineBySemicolon[2]);
                            }
                        }
                        else
                        {
                            RecommendedWrongChoicePointsTB.Text = "";
                            SelectedWrongChoicePointsTB.Text = "";
                            SelectedWrongChoicePointsRB.Enabled = false;
                            RecommendedWrongChoicePointsRB.Enabled = false;
                            RecommendedWrongChoicePointsRB.Checked = true;
                            SelectedWrongChoicePointsTB.ReadOnly = true;
                        }

                        if(questionType == 5)
                        {
                            RecommendedWrongChoicePointsTB.Text = "";
                            SelectedWrongChoicePointsTB.Text = "";
                            SelectedWrongChoicePointsRB.Enabled = false;
                            RecommendedWrongChoicePointsRB.Enabled = false;
                            RecommendedWrongChoicePointsRB.Checked = true;
                            SelectedWrongChoicePointsTB.ReadOnly = true;
                        }
                    }

                    if (splitImportedFileLineBySemicolon[1] != "N/A")
                    {
                        questionPoints += int.Parse(splitImportedFileLineBySemicolon[1]);
                    }
                }

                if(!itemRecordExists)
                {
                    SubquestionPointsLabel.Text = "Počet bodů za podotázku: N/A";
                    SubquestionPointsTB.Text = "N/A";
                    string itemPointsText = subitemIdentifier + ";N/A" + Environment.NewLine;
                    File.AppendAllText(itemParentPath + "\\Points.txt", itemPointsText);
                }
            }

            if(!fileExists || !itemRecordExists || undecidedPointsInFile)
            {
                QuestionPointsLabel.Text = "Počet bodů za otázku: N/A";
                CorrectChoicePointsTB.Text = "N/A";
            }
            else
            {
                if(!isTeacherReviewingDeliveryResult && isTeacherEditingQuestion)
                {
                    QuestionPointsLabel.Text = "Počet bodů za otázku: " + questionPoints.ToString();
                    CorrectChoicePointsTB.Text = GetCorrectChoicePoints().ToString();
                }
            }
        }

        public void SaveQuestionPoints()
        {
            string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier;

            bool isNumber = int.TryParse(SubquestionPointsTB.Text, out _);
            bool isWrongChoicePointsNumber = double.TryParse(SelectedWrongChoicePointsTB.Text, out _);
            if (!isNumber || (!isWrongChoicePointsNumber && SelectedWrongChoicePointsRB.Checked))
            {
                MessageBox.Show("Chyba: je nutné zadat číslo.", "Chybný počet bodů", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                bool performSave = true;
                string warningText = "";
                string warningTitle = "";
                if(SelectedWrongChoicePointsTB.Text != "" && Math.Abs(double.Parse(SelectedWrongChoicePointsTB.Text)) > subquestionPoints)
                {
                    warningText = "Varování: za špatný výběr bude studentovi odečteno více bodů, než kolik může dostat za otázku. Chcete pokračovat?";
                    warningTitle = "Varování - počet bodů";
                    DialogResult result = MessageBox.Show(warningText, warningTitle, MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    if (result == DialogResult.No)
                    {
                        performSave = false;
                    }
                }

                if(SelectedWrongChoicePointsTB.Text != "" && double.Parse(SelectedWrongChoicePointsTB.Text) > 0)
                {
                    performSave = false;
                    warningText = "Chyba: za špatnou volbu nemůže být udělen kladný počet bodů.";
                    warningTitle = "Chyba - kladný počet bodů";
                    MessageBox.Show(warningText, warningTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }

                if (SubquestionPointsTB.Text != "" && double.Parse(SubquestionPointsTB.Text) < 0)
                {
                    performSave = false;
                    warningText = "Chyba: za správnou odpověď nemůže být udělen záporný počet bodů";
                    warningTitle = "Chyba - záporný počet bodů";
                    MessageBox.Show(warningText, warningTitle, MessageBoxButtons.OK, MessageBoxIcon.Error);
                }

                if (performSave)
                {
                    subquestionPoints = int.Parse(SubquestionPointsTB.Text);
                    double wrongChoicePoints;
                    if (RecommendedWrongChoicePointsRB.Checked || !RecommendedWrongChoicePointsRB.Enabled)
                    {
                        wrongChoicePoints = GetCorrectChoicePoints() * (-1);
                        SelectedWrongChoicePointsTB.ReadOnly = true;
                        SelectedWrongChoicePointsTB.Text = "";
                    }
                    else
                    {
                        wrongChoicePoints = double.Parse(SelectedWrongChoicePointsTB.Text);
                    }

                    if (questionType == 5)
                    {
                        wrongChoicePoints = 0;
                    }

                    string[] importedFileLines = File.ReadAllLines(itemParentPath + "\\Points.txt");
                    string fileLinesToExport = "";
                    for (int i = 0; i < importedFileLines.Length; i++)
                    {
                        string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                        if (splitImportedFileLineBySemicolon[0] == subitemIdentifier)
                        {
                            importedFileLines[i] = subitemIdentifier + ";" + SubquestionPointsTB.Text + ";" + Math.Round(wrongChoicePoints, 2);
                        }
                        fileLinesToExport += importedFileLines[i] + "\n";
                    }
                    File.WriteAllText(itemParentPath + "\\Points.txt", fileLinesToExport);
                    MessageBox.Show("Počet bodů u podotázky byl úspešně změněn.", "Počet bodů změněn", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    LoadQuestionPoints();
                }
            }
        }

        private void ReturnButton_Click(object sender, EventArgs e)
        {
            new TestForm((testNameIdentifier, testNumberIdentifier), isTeacherEditingQuestion, deliveryExecutionIdentifier, studentIdentifier, isTeacherReviewingDeliveryResult).Show();
            Hide();
        }

        private void SubitemCB_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (subitemsAdded)
            {
                LoadItemInfo();
            }
            if (SubitemCB.SelectedIndex != -1)
            {
                subitemIdentifier = responseIdentifierArray[SubitemCB.SelectedIndex];
                ResponseIdentifierLabel.Text = "Identifikátor podotázky: " + responseIdentifierArray[SubitemCB.SelectedIndex];
            }
            if(subitemsAdded && !isTeacherEditingQuestion)
            {
                LoadDeliveryExecutionInfo();
            }
            if(subitemsAdded && isTeacherReviewingDeliveryResult)
            {
                LoadDeliveryExecutionInfoToEdit();
            }
        }

        private void SaveSubquestionPointsButton_Click(object sender, EventArgs e)
        {
            SaveQuestionPoints();
        }

        private void SaveStudentsPointsButton_Click(object sender, EventArgs e)
        {
            bool isDecimal = double.TryParse(StudentsPointsTB.Text, out _);
            if(!isDecimal)
            {
                MessageBox.Show("Chyba: je nutné zadat číslo.", "Chybný počet bodů", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else if(undecidedPointsInFile)
            {
                MessageBox.Show("Chyba: není možné upravit počet bodů studenta. Nejprve je nutné určit počet obdržených bodů za otázku.", "Počet bodů za otázku není určený", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                string[] resultsFileLines = File.ReadAllLines("C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + "Results.txt");
                string resultsToFile = "";

                for (int i = 0; i < resultsFileLines.Length; i++)
                {
                    string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                    if (splitResultsFileLineBySemicolon[0] != itemNameIdentifier)
                    {
                        resultsToFile += resultsFileLines[i] + "\n";
                    }
                    else
                    {
                        if (amountOfSubitems > 1)
                        {
                            resultsToFile += itemNameIdentifier;

                            for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                            {
                                resultsToFile += ";";
                                if(j-1 != SubitemCB.SelectedIndex)
                                {
                                    resultsToFile += splitResultsFileLineBySemicolon[j];
                                }
                                else
                                {
                                    resultsToFile += StudentsPointsTB.Text;
                                }
                            }

                            resultsToFile += "\n";
                        }
                        else
                        {
                            resultsToFile += itemNameIdentifier + ";" + StudentsPointsTB.Text + "\n";
                        }
                    }
                }

                File.WriteAllText("C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + "Results.txt", resultsToFile);
                MessageBox.Show("Počet získaných bodů byl u studenta úspěšně upraven.", "Počet bodů změněn", MessageBoxButtons.OK, MessageBoxIcon.Information);
                importedReceivedPointsArray.Clear();
                LoadDeliveryExecutionInfo();
                LoadDeliveryExecutionInfoToEdit();
            }
        }

        private void SelectedWrongChoicePointsRB_CheckedChanged(object sender, EventArgs e)
        {
            SelectedWrongChoicePointsTB.ReadOnly = !SelectedWrongChoicePointsRB.Checked;
            if(!SelectedWrongChoicePointsRB.Checked)
            {
                SelectedWrongChoicePointsTB.Text = "";
            }
        }
    }
}
