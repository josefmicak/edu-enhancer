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

namespace TAO_Enhancer
{
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
        List<string> responseValueArray = new List<string>();//TODO 5: Tohle by asi mohl být Tuple s responseIdentifierArray
        int questionType = 0;
        int amountOfSubitems = 0;
        bool subitemsAdded = false;
        List<bool> includesImage = new List<bool>();
        int subquestionPoints = 0;
        int questionPoints = 0;

        bool isTeacher = true;
        string deliveryExecutionIdentifier = "";
        string studentIdentifier = "";
        List<double> studentsReceivedPointsArray = new List<double>();
        bool deliveryExecutionFileCreated;
        List<double> importedReceivedPointsArray = new List<double>();

        public ItemForm(string testNameID, string itemNameID, string itemNumberID, string testNumberID, bool requestOrigin, string attemptIdentifier, string studentID, bool attemptFileCreated)
        {
            InitializeComponent();
            testNameIdentifier = testNameID;
            itemNameIdentifier = itemNameID;
            itemNumberIdentifier = itemNumberID;
            testNumberIdentifier = testNumberID;
            deliveryExecutionFileCreated = attemptFileCreated;
            LoadItemInfo();

            if (!requestOrigin)//student
            {
                deliveryExecutionIdentifier = attemptIdentifier;
                isTeacher = false;
                studentIdentifier = studentID;
                LoadDeliveryExecutionInfo();
                ModifyQuestionGB.Visible = false;
            }
        }

        public void LoadItemInfo()
        {
            NumberIdentifierLabel.Text = "Číselný identifikátor: " + itemNumberIdentifier;
            NameIdentifierLabel.Text = "Jmenný identifikátor: " + itemNameIdentifier;

            bool includesQuestion = false;

            ResetLoadedItemInfo();

            DoesSubitemIncludeImage();

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

            /*
            if (questionType == 3 || questionType == 4)
            {
                GetChoiceIdentifierValues();
            }
            */
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

                if (includesImage[currentSubitem])
                {
                    if (xmlReader.Name == "div" && xmlReader.AttributeCount == 0 && xmlReader.NodeType != XmlNodeType.EndElement)//TODO 3: Předělat (?), div je potomkem prompt, viz TODO 2
                    {
                        QuestionLabel.Text = "Otázka: " + xmlReader.ReadElementContentAsString();
                    }
                    if (xmlReader.Name == "img")
                    {
                        QuestionImage.ImageLocation = ("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier + "\\" + xmlReader.GetAttribute("src"));
                    }
                }
                else
                {
                    if (questionType == 7)
                    {
                        if (xmlReader.Name == "p")
                        {
                            //TODO 9: Inline interactions nezprovozněno
                            QuestionLabel.Text = "Otázka: " + xmlReader.ReadString();
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

                if(!includesQuestion && !includesImage[currentSubitem])
                {
                    QuestionLabel.Text = "Otázka nebyla vyplněna.";//TODO: Pokud obsahuje obrázek a neobsahuje text, tohle se nevyvolá
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
                    if(attemptIdentifierSplitByUnderscore[2] == deliveryExecutionIdentifier)
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
                                if(questionType == 3 || questionType == 4)
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
                                else if(questionType == 5)
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

            string studentsAnswerToLabel = "";
            int answerNumber = 0;
            bool studentAnsweredQustion = false;
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
                        else
                        {
                            studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + "\n";
                        }
                    }
                }

                if(studentsAnswers[i] != "")
                {
                    studentAnsweredQustion = true;
                }
            }

            if(questionType == 5)
            {
                studentsAnswerToLabel = studentsAnswers[0];
            }

            if (!studentAnsweredQustion)
            {
                studentsAnswerToLabel = "Nevyplněno";
            }

            StudentsAnswerLabel.Text = "Vaše odpověď: \n" + studentsAnswerToLabel;

            double studentsReceivedPoints = 0;
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
                    case int n when (n == 1 || n == 6):
                        bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctChoiceArray, studentsAnswers);
                        if (areStudentsAnswersCorrect)
                        {
                            studentsReceivedPoints = subquestionPoints;
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

                        studentsReceivedPoints -= Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * ((double)subquestionPoints / (double)correctChoiceArray.Count);
                        break;
                    case int n when (n == 3 || n == 4):
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

                        studentsReceivedPoints -= Math.Abs(studentsAnswers.Count - studentsCorrectAnswers) * ((double)subquestionPoints / (double)correctChoiceArray.Count);
                        break;
                }
            }

            switch (questionType)
            {
                case int n when (n == 1 || n == 6):
                    bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctChoiceArray, studentsAnswers);
                    if (!areStudentsAnswersCorrect)
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Incorrect;
                    }
                    else
                    {
                        isAnswerCorrect = StudentsAnswerCorrectness.Correct;
                        studentsReceivedPoints = subquestionPoints;
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
                case int n when (n == 3 || n == 4):
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

            if (studentsReceivedPoints < 0)//TODO: Záporné body
            {
                studentsReceivedPoints = 0;
            }

            if (subquestionPoints == -1)
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

                    if(responseIdentifierArray.Count - 1 > responseValueArray.Count)
                    {
                        responseValueArray.Add("Otázka nebyla vyplněna.");
                    }
                }

                if(responseIdentifierArray.Count > 0)
                {
                    if(includesImage.Count == 0)
                    {//TODO: Ošetřit includesImage tak, aby ověřil přítomnost obrázku i bez promptu
                        MessageBox.Show("Chyba: otázka nemá pravděpodobně zadaný žádný text. Otázku nelze načíst.", "Chyba", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        Environment.Exit(0);
                    }
                    else
                    {
                        if (includesImage[responseIdentifierArray.Count - 1])
                        {
                            if (xmlReader.Name == "div" && xmlReader.AttributeCount == 0 && xmlReader.NodeType != XmlNodeType.EndElement)//TODO 3: Předělat (?), div je potomkem prompt, viz TODO 2
                            {
                                string responseValue = xmlReader.ReadElementContentAsString();
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
                        }
                    }
                }
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

        public void DoesSubitemIncludeImage()
        {
            bool imageFound = false;
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if(xmlReader.Name == "prompt" && xmlReader.NodeType == XmlNodeType.EndElement && !imageFound)
                {
                    includesImage.Add(false);
                }

                if (xmlReader.Name == "img")
                {
                    imageFound = true;
                    includesImage.Add(true);
                }

                if (xmlReader.Name == "prompt" && xmlReader.NodeType == XmlNodeType.EndElement && imageFound)
                {
                    imageFound = false;
                }
            }
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
                    {//TODO 10: Problém s tím, když je jedna možnost ve více párech
                        questionType = 3;//Typ otázky = spojování párů
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "directedPair")
                    {//TODO: co když je jen jedno správně?
                        questionType = 4;//Typ otázky = více otázek (tabulka); více odpovědí může být správně
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "string")
                    {//TODO 11: Přidat typ otázky, kde je volná otázka ale daná odpověď
                        questionType = 5;//Typ otázky = volná odpověď;
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        singleCorrectAnswer = true;
                    }
                }

                if (singleCorrectAnswer)
                {
                    if (xmlReader.Name == "simpleChoice")
                    {
                        questionType = 6;//Typ otázky = výběr z více možností (abc), jen jedna odpověď je správně
                    }
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
            }
        }

        public void GetChoiceIdentifierValues()
        {
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "simpleChoice" || xmlReader.Name == "simpleAssociableChoice")
                {
                    string choiceIdentifier = xmlReader.GetAttribute("identifier");
                    string choiceValue = xmlReader.ReadElementContentAsString();
                    choiceIdentifierValueTuple.Add((choiceIdentifier, choiceValue));
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
                    if (xmlReader.Name == "simpleChoice" || xmlReader.Name == "simpleAssociableChoice")
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
                PossibleAnswerLabel.Text = "Jedná se o otevřenou otázku, neobsahuje výběr z možností, odpovědi je nutné ověřit manuálně.\n" + possibleAnswers;
            }
            else
            {
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

        public void LoadQuestionPoints()
        {
            if(amountOfSubitems > 1)
            {
                subitemIdentifier = responseIdentifierArray[SubitemCB.SelectedIndex];
            }

            bool fileExists = false;
            bool itemRecordExists = false;
            bool undecidedPointsInFile = false;
            questionPoints = 0;//
            string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier;
            foreach (var file in Directory.GetFiles(itemParentPath))
            {
                if(Path.GetFileName(file) == "Points.txt")
                {
                    fileExists = true;
                }
            }

            if(!fileExists)
            {
                string itemPointsText = subitemIdentifier + ";N/A" + Environment.NewLine;
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
                            subquestionPoints = -1;
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
            }
            else
            {
             //   QuestionPointsLabel.Text = "Počet bodů za otázku: " + questionPoints.ToString();
            }
        }

        public void SaveQuestionPoints()
        {
            string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier;

            bool isNumber = int.TryParse(SubquestionPointsTB.Text, out _);
            if(!isNumber)
            {
                MessageBox.Show("Chyba: je nutné zadat číslo.", "Chybný počet bodů", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                string[] importedFileLines = File.ReadAllLines(itemParentPath + "\\Points.txt");
                string fileLinesToExport = "";
                for (int i = 0; i < importedFileLines.Length; i++)
                {
                    string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                    if (splitImportedFileLineBySemicolon[0] == subitemIdentifier)
                    {
                        importedFileLines[i] = subitemIdentifier + ";" + SubquestionPointsTB.Text;
                    }
                    fileLinesToExport += importedFileLines[i] + "\n";
                }
                File.WriteAllText(itemParentPath + "\\Points.txt", fileLinesToExport);
                MessageBox.Show("Počet bodů u podotázky byl úspešně změněn.", "Počet bodů změněn", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }

            LoadQuestionPoints();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new TestForm((testNameIdentifier, testNumberIdentifier), isTeacher, deliveryExecutionIdentifier, studentIdentifier).Show();
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
            if(subitemsAdded && !isTeacher)
            {
                LoadDeliveryExecutionInfo();
            }
        }

        private void SaveSubquestionPointsButton_Click(object sender, EventArgs e)
        {
            SaveQuestionPoints();
        }
    }
}
