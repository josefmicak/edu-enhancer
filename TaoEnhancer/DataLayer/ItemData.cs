using Common;
using Common.Class;
using System.Xml;

namespace DataLayer
{
    public class ItemData
    {
        private TestData testData = new TestData();

        /*----- Enumerations -----*/
        enum StudentsAnswerCorrectness
        {
            Correct,
            Incorrect,
            PartiallyCorrect,
            Unknown
        }

        /*----- Auxiliary functions -----*/
        // #DELETE
        private (int, bool) GetTestPoints(List<(string, string, string, string, int, bool)> itemParameters)
        {
            int testPoints = 0;
            bool testPointsDetermined = true;

            for (int i = 0; i < itemParameters.Count; i++)
            {
                testPoints += itemParameters[i].Item5;
                if (!itemParameters[i].Item6)
                {
                    testPointsDetermined = false;
                }
            }

            return (testPoints, testPointsDetermined);
        }

        private string GetChoiceValue(string choiceIdentifier, List<(string, string)> choiceIdentifierValueTuple)
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

        /*----- File functions -----*/
        public string[] LoadPoints(string testNameIdentifier, string itemNumberIdentifier)
        {
            string file = Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier);
            return File.ReadAllLines(file);
        }

        private void SavePoints(string testNameIdentifier, string itemNumberIdentifier, string fileLinesToExport)
        {
            string file = Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier);
            File.WriteAllText(file, fileLinesToExport);
        }

        private void AppendPoints(string testNameIdentifier, string itemNumberIdentifier, string fileLinesToExport)
        {
            string file = Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier);
            File.AppendAllText(file, fileLinesToExport);
        }

        // HomeController.cs (down)
        private void UpdatePoints(string testNameIdentifier, string itemNumberIdentifier, string responseIdentifier, int correctChoicePoints, double wrongChoicePoints)
        {
            string[] importedFileLines = LoadPoints(testNameIdentifier, itemNumberIdentifier);
            string fileLinesToExport = "";
            for (int i = 0; i < importedFileLines.Length; i++)
            {
                string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                if (splitImportedFileLineBySemicolon[0] == responseIdentifier)
                {
                    importedFileLines[i] = responseIdentifier + ";" + correctChoicePoints + ";" + Math.Round(wrongChoicePoints, 2);
                }
                fileLinesToExport += importedFileLines[i] + "\n";
            }
            SavePoints(testNameIdentifier, itemNumberIdentifier, fileLinesToExport);
        }

        /*private List<(string, string, string, string, int, bool)> LoadItemInfo(string testNameIdentifier)
        {
            // #CHANGED
            //string title = "";
            //string label = "";
            //string itemNumberIdentifier = "";
            //string itemNameIdentifier = "";
            int i = 0;
            List<(string, string, string, string, int, bool)> itemParametersTemp = new List<(string, string, string, string, int, bool)>();
            List<(string, string, string, string)> questionList = testData.LoadQuestions(testNameIdentifier);
            // #CHANGED
            //bool testPointsDetermined = false;
            //int testPoints = 0;

            foreach (var directory in Directory.GetDirectories(Settings.GetTestItemsPath(testNameIdentifier)))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string[] fileSplitBySlash = file.Split(@"\");
                    if (fileSplitBySlash[fileSplitBySlash.Length - 1] != "qti.xml")
                    {
                        continue;
                    }
                    else
                    {
                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                            {
                                if (xmlReader.Name == "assessmentItem")
                                {
                                    for (int j = 0; j < questionList.Count; j++)
                                    {
                                        if (questionList[j].Item4 == xmlReader.GetAttribute("identifier"))
                                        {
                                            string itemNumberIdentifier = questionList[j].Item4;
                                            string itemNameIdentifier = questionList[j].Item3;
                                            string title = xmlReader.GetAttribute("title");
                                            string label = xmlReader.GetAttribute("label");
                                            (int questionPoints, bool questionPointsDetermined) = LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier);

                                            itemParametersTemp.Add((itemNumberIdentifier, itemNameIdentifier, title, label, questionPoints, questionPointsDetermined));
                                            i++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //pole nyní může být ve špatném pořadí kvůli jinému pořadí v rámci qti.xml celého testu a složek ve složce items - oprava
            List<(string, string, string, string, int, bool)> itemParameters = new List<(string, string, string, string, int, bool)>();
            for (int k = 0; k < questionList.Count; k++)
            {
                for (int l = 0; l < itemParametersTemp.Count; l++)
                {
                    if (questionList[k].Item4 == itemParametersTemp[l].Item1)
                    {
                        itemParameters.Add(itemParametersTemp[l]);
                    }
                }
            }

            return itemParameters;
        }*/

        /*private (int, bool) LoadQuestionPoints(string testNameIdentifier, string itemNumberIdentifier)
        {
            // #CHANGED
            if (File.Exists(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier)))
            {
                bool questionPointsDetermined = true;
                int questionPoints = 0;

                string[] importedFileLines = File.ReadAllLines(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier));
                for (int j = 0; j < importedFileLines.Length; j++)
                {
                    string[] splitImportedFileLineBySemicolon = importedFileLines[j].Split(";");

                    if (splitImportedFileLineBySemicolon[1] == "N/A")
                    {
                        questionPointsDetermined = false;
                    }
                    else
                    {
                        questionPoints += int.Parse(splitImportedFileLineBySemicolon[1]);
                    }
                }

                return (questionPoints, questionPointsDetermined);
            }
            return (0, false);
        }*/

        private (List<string>, List<string>, int) GetResponseIdentifiers(string testNameIdentifier, string itemNumberIdentifier, int amountOfSubitems)
        {
            List<string> responseIdentifierArray = new List<string>();
            List<string> responseValueArray = new List<string>();
            string questionText = "";
            int amountOfAddedFaultyQuestions = 0;
            int errorMessageNumber = 0;
            List<(bool, string, string)> includesImage = new List<(bool, string, string)>();

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
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
                    int questionType = GetQuestionType(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems);
                    if (questionType == 7 || questionType == 8)
                    {
                        amountOfAddedFaultyQuestions++;
                        string faultyQuestionValue = GetFaultyQuestionValue(testNameIdentifier, itemNumberIdentifier, amountOfAddedFaultyQuestions);
                        responseValueArray.Add(faultyQuestionValue);
                    }

                    (int amountOfImages, questionText, _, includesImage, errorMessageNumber) = SubitemImages(testNameIdentifier, itemNumberIdentifier, responseIdentifier, includesImage);

                    if (responseIdentifierArray.Count - 1 > responseValueArray.Count)
                    {
                        responseValueArray.Add("Otázka nebyla vyplněna.");
                    }
                }

                if (responseIdentifierArray.Count > 0)
                {
                    if (includesImage.Count == 0)
                    {
                        errorMessageNumber = 2;
                    }
                    else
                    {
                        if (xmlReader.Name == "gapMatchInteraction")
                        {
                            using (var innerReader = xmlReader.ReadSubtree())
                            {
                                while (innerReader.Read())
                                {

                                    if (innerReader.Name == "p")
                                    {
                                        string gapText = innerReader.ReadInnerXml();
                                        questionText = "";
                                        bool addText = true;
                                        int gapCounter = 1;

                                        for (int i = 0; i < gapText.Length; i++)
                                        {
                                            if (gapText[i] == '<')
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
                                            if (addText)
                                            {
                                                questionText += gapText[i];
                                            }
                                        }
                                        responseValueArray.Add(questionText);
                                    }
                                }
                            }
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
                                    string responseValue = questionText;
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
            }

            for (int i = 0; i < includesImage.Count; i++)
            {
                if (includesImage[i].Item1 && includesImage[i].Item2 == "")
                {
                    includesImage[i] = (includesImage[i].Item1, responseValueArray[i], includesImage[i].Item3);
                }

                if (includesImage[i].Item1 && includesImage[i].Item2 == "")
                {
                    XmlReader xmlReaderCorrection = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
                    while (xmlReaderCorrection.Read())
                    {
                        if (xmlReaderCorrection.Name == "choiceInteraction")
                        {
                            if (xmlReaderCorrection.GetAttribute("responseIdentifier") != responseIdentifierArray[i])
                            {
                                xmlReaderCorrection.Skip();
                            }
                        }
                        if (xmlReaderCorrection.Name == "gapMatchInteraction")
                        {
                            xmlReaderCorrection.Skip();
                        }
                        if (xmlReaderCorrection.Name == "prompt")
                        {
                            string promptQuestionText = xmlReaderCorrection.ReadInnerXml();
                            int firstStartTag = promptQuestionText.IndexOf('<');
                            int lastEndTag = promptQuestionText.LastIndexOf('>');
                            string questionTextNew = promptQuestionText.Substring(0, firstStartTag) + promptQuestionText.Substring(1 + lastEndTag);
                            responseValueArray[i] = questionTextNew;
                            includesImage[i] = (includesImage[i].Item1, responseValueArray[i], includesImage[i].Item3);
                            break;
                        }
                    }
                }
            }

            return (responseIdentifierArray, responseValueArray, errorMessageNumber);
        }

        private int GetQuestionType(string testNameIdentifier, string itemNumberIdentifier, string responseIdentifierCorrection, int amountOfSubitems)
        {
            bool unknownQuestionType = false;
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            bool singleCorrectAnswer = false;//questionType = 6 nebo 7; jediná správná odpověď
            int questionType = 0;

            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes && xmlReader.Name == "responseDeclaration")
                {
                    if (amountOfSubitems > 1)
                    {
                        string responseIdentifierTemp = xmlReader.GetAttribute("identifier");
                        if (responseIdentifierTemp != null && responseIdentifierTemp != responseIdentifierCorrection)
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
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "integer")
                    {
                        questionType = 10;//Typ otázky = posuvník; jedna správná odpověď (číslo)
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "identifier")
                    {
                        singleCorrectAnswer = true;
                    }
                }

                if (xmlReader.Name == "gapMatchInteraction")
                {
                    if (amountOfSubitems > 1)
                    {
                        string responseIdentifierGap = xmlReader.GetAttribute("responseIdentifier");
                        if (responseIdentifierGap == responseIdentifierCorrection)
                        {
                            questionType = 9;
                        }
                    }
                    else
                    {
                        questionType = 9;
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

                string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                if (responseIdentifier != null && responseIdentifier != responseIdentifierCorrection)
                {
                    xmlReader.Skip();
                }
                if (xmlReader.NodeType != XmlNodeType.EndElement && (xmlReader.Name == "hottextInteraction" || xmlReader.Name == "mediaInteraction" || xmlReader.Name == "hotspotInteraction"
                    || xmlReader.Name == "customInteraction" || xmlReader.Name == "graphicOrderInteraction" || xmlReader.Name == "graphicAssociateInteraction" || xmlReader.Name == "graphicGapMatchInteraction"
                    || xmlReader.Name == "selectPointInteraction"))
                {
                    unknownQuestionType = true;
                    questionType = 0;//Typ otázky = neznámá otázka
                }
            }

            if (singleCorrectAnswer && questionType == 0 && !unknownQuestionType)
            {
                questionType = 7;//Typ otázky = výběr z více možností (dropdown), jen jedna odpověď je správně
            }
            return questionType;
        }

        private string GetFaultyQuestionValue(string testNameIdentifier, string itemNumberIdentifier, int amountOfAddedFaultyQuestions)
        {
            int amountOfFaultyQuestions = 0;

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "gapMatchInteraction")
                {
                    xmlReader.Skip();
                }

                if (xmlReader.Name == "p" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    amountOfFaultyQuestions++;
                    if (amountOfAddedFaultyQuestions != amountOfFaultyQuestions)
                    {
                        xmlReader.Skip();
                    }
                    else
                    {
                        string inlineChoiceInteractionLine = xmlReader.ReadInnerXml();
                        int firstStartTag = inlineChoiceInteractionLine.IndexOf('<');
                        int lastEndTag = inlineChoiceInteractionLine.LastIndexOf('>');
                        string questionText = inlineChoiceInteractionLine.Substring(0, firstStartTag) + "(DOPLŇTE)" + inlineChoiceInteractionLine.Substring(1 + lastEndTag);
                        return questionText;
                    }
                }
            }

            return "Při přidávání otázky nastala neočekávaná chyba";
        }

        private (int, string, string, List<(bool, string, string)>, int) SubitemImages(string testNameIdentifier, string itemNumberIdentifier, string responseIdentifier, List<(bool, string, string)> includesImage)
        {
            int amountOfImages = 0;
            string questionText = "";
            string imageSource = "";

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element)
                {
                    var name = xmlReader.Name;
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction")
                    {
                        if (xmlReader.GetAttribute("responseIdentifier") != responseIdentifier)
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
                                        while (innerReaderNext.Read())
                                        {
                                            if (innerReaderNext.Name == "img")
                                            {
                                                imageSource = innerReaderNext.GetAttribute("src");
                                                amountOfImages++;
                                            }
                                            if (innerReaderNext.Name == "div")
                                            {
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

            int errorMessageNumber = 0;
            if (amountOfImages > 1)
            {
                errorMessageNumber = 1;
            }
            else if (amountOfImages == 1)
            {
                includesImage.Add((true, questionText, imageSource));
            }
            else
            {
                includesImage.Add((false, questionText, imageSource));
            }

            return (amountOfImages, questionText, imageSource, includesImage, errorMessageNumber);
        }

        private (string, int, int, bool, double, string, string, List<string>, List<string>, List<string>, List<string>) LoadSubitemParameters(string testNameIdentifier, string itemNumberIdentifier, string responseIdentifier, int amountOfSubitems, List<string> responseIdentifierArray, List<string> responseValueArray)//načte parametry dané podotázky
        {
            int questionType = GetQuestionType(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems);
            (int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints) = GetSubquestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems, questionType);
            List<(bool, string, string)> includesImage = new List<(bool, string, string)>();
            //TODO: Zde je odkaz na můj web, protože v ASP.NET nejde zobrazit obrázek z pevného disku. Ostatní soubory se zobrazují z pevného disku.
            string imageSource = "";
            if (SubitemImages(testNameIdentifier, itemNumberIdentifier, responseIdentifier, includesImage).Item3 != "")
            {
                imageSource = Settings.GetTestItemImagePath(testNameIdentifier, itemNumberIdentifier, SubitemImages(testNameIdentifier, itemNumberIdentifier, responseIdentifier, includesImage).Item3);
            }
            string subitemText = "";

            for (int i = 0; i < responseIdentifierArray.Count; i++)
            {
                if (responseIdentifier == responseIdentifierArray[i])
                {
                    subitemText = responseValueArray[i];
                }
            }

            (List<string> possibleAnswerArray, List<string> subquestionArray) = GetPossibleAnswerList(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems, questionType);
            (List<string> correctChoiceArray, List<string> correctAnswerArray) = GetCorrectAnswerList(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems, questionType);

            return (responseIdentifier, questionType, subquestionPoints, subquestionPointsDetermined, wrongChoicePoints, imageSource, subitemText, possibleAnswerArray, subquestionArray, correctChoiceArray, correctAnswerArray);
        }

        private (int, bool, double) GetSubquestionPoints(string testNameIdentifier, string itemNumberIdentifier, string responseIdentifier, int amountOfSubitems, int questionType)
        {
            bool fileExists = false;
            int subquestionPoints = 0;
            bool subquestionPointsDetermined = true;
            double wrongChoicePoints = 0;

            if (File.Exists(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier)))
            {
                string[] importedFileLines = LoadPoints(testNameIdentifier, itemNumberIdentifier);
                for (int i = 0; i < importedFileLines.Length; i++)
                {
                    string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                    /*  if (splitImportedFileLineBySemicolon[1] == "N/A" || importedFileLines.Length != amountOfSubitems)
                      {
                          subquestionPointsDetermined = false;
                      }*/

                    if (splitImportedFileLineBySemicolon[0] == responseIdentifier)
                    {
                        if (splitImportedFileLineBySemicolon[1] != "N/A")
                        {
                            subquestionPoints = int.Parse(splitImportedFileLineBySemicolon[1]);
                        }
                        else
                        {
                            subquestionPointsDetermined = false;
                        }

                        if (splitImportedFileLineBySemicolon.Length > 2 && splitImportedFileLineBySemicolon[2] != "N/A")
                        {
                            wrongChoicePoints = double.Parse(splitImportedFileLineBySemicolon[2]);
                        }
                    }
                }
            }
            else
            {
                subquestionPointsDetermined = false;
                string fileLinesToExport = responseIdentifier + ";N/A;N/A" + Environment.NewLine;
                SavePoints(testNameIdentifier, itemNumberIdentifier, fileLinesToExport);
            }

            return (subquestionPoints, subquestionPointsDetermined, wrongChoicePoints);
        }

        private (List<string>, List<string>) GetPossibleAnswerList(string testNameIdentifier, string itemNumberIdentifier, string selectedResponseIdentifier, int amountOfSubitems, int questionType)
        {
            List<string> possibleAnswerArray = new List<string>();
            List<string> subquestionArray = new List<string>();
            int simpleMatchSetCounter = 0;

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (amountOfSubitems > 1)
                {
                    string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                    if (responseIdentifier != null && responseIdentifier != selectedResponseIdentifier)
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
                else if (questionType == 10)
                {
                    if (xmlReader.Name == "sliderInteraction" && xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        string lowerBound = xmlReader.GetAttribute("lowerBound");
                        string upperBound = xmlReader.GetAttribute("upperBound");
                        possibleAnswerArray.Add(lowerBound + " - " + upperBound);
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

            if (questionType == 7)
            {
                XmlReader xmlReaderInlineChoice = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
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
            }

            return (possibleAnswerArray, subquestionArray);
        }

        private (List<string>, List<string>) GetCorrectAnswerList(string testNameIdentifier, string itemNumberIdentifier, string selectedResponseIdentifier, int amountOfSubitems, int questionType)
        {
            List<string> correctChoiceArray = new List<string>();
            List<string> correctAnswerArray = new List<string>();
            //string correctAnswer = "";

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (amountOfSubitems > 1 && xmlReader.Name == "responseDeclaration")
                {
                    string responseIdentifier = xmlReader.GetAttribute("identifier");
                    if (responseIdentifier != null && responseIdentifier != selectedResponseIdentifier)
                    {
                        xmlReader.Skip();
                    }
                }

                if (questionType == 3 || questionType == 4)
                {
                    if (xmlReader.Name == "correctResponse")
                    {
                        using (var innerReader = xmlReader.ReadSubtree())
                        {
                            bool whileLoopBool;
                            while (whileLoopBool = innerReader.ReadToFollowing("value"))
                            {
                                string value = innerReader.ReadString();
                                string[] orderedCorrectChoices = value.Split(' ');
                                correctChoiceArray.Add(orderedCorrectChoices[1]);
                                correctChoiceArray.Add(orderedCorrectChoices[0]);
                            }
                        }
                    }
                }
                else if (questionType == 8)
                {
                    if (xmlReader.Name == "correctResponse")
                    {
                        using (var innerReader = xmlReader.ReadSubtree())
                        {
                            bool whileLoopBool;
                            while (whileLoopBool = innerReader.ReadToFollowing("value"))
                            {
                                string value = innerReader.ReadString();
                                correctAnswerArray.Add(value);
                            }
                        }
                    }
                }
                else if (questionType == 9)
                {
                    if (xmlReader.Name == "correctResponse")
                    {
                        using (var innerReader = xmlReader.ReadSubtree())
                        {
                            bool whileLoopBool;
                            while (whileLoopBool = innerReader.ReadToFollowing("value"))
                            {
                                string value = innerReader.ReadString();
                                string[] orderedCorrectChoices = value.Split(' ');
                                correctChoiceArray.Add(orderedCorrectChoices[0]);
                                correctChoiceArray.Add(orderedCorrectChoices[1]);
                            }
                        }
                    }

                    if (xmlReader.Name == "gapText")
                    {
                        for (int i = 0; i < correctChoiceArray.Count; i++)
                        {
                            if (i % 2 == 1)
                            {
                                continue;
                            }

                            if (xmlReader.GetAttribute("identifier") == correctChoiceArray[i])
                            {
                                correctAnswerArray.Add(xmlReader.ReadElementContentAsString());
                            }
                        }
                    }
                }
                else if (questionType == 10)
                {
                    if (xmlReader.Name == "correctResponse")
                    {
                        using (var innerReader = xmlReader.ReadSubtree())
                        {
                            if (innerReader.ReadToFollowing("value"))
                            {
                                string sliderStudentAnswer = innerReader.ReadElementContentAsString();
                                correctChoiceArray.Add(sliderStudentAnswer);
                                correctAnswerArray.Add(sliderStudentAnswer);
                            }
                        }
                    }
                }
                else
                {
                    if (xmlReader.Name == "correctResponse")
                    {
                        using (var innerReader = xmlReader.ReadSubtree())
                        {
                            bool whileLoopBool;
                            while (whileLoopBool = innerReader.ReadToFollowing("value"))
                            {
                                string value = innerReader.ReadString();
                                correctChoiceArray.Add(value);
                                correctAnswerArray.Add(" ");//placeholder
                            }
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

            if (questionType == 7)
            {
                correctAnswerArray.Clear();

                XmlReader xmlReaderInlineChoice = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
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
                                            for (int i = 0; i < correctChoiceArray.Count; i++)
                                            {
                                                if (innerReaderNext.GetAttribute("identifier") == correctChoiceArray[i])
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

            if (questionType == 3 || questionType == 4)
            {
                foreach (string answer in correctChoiceArray)
                {
                    correctAnswerArray.Add(GetChoiceValue(answer, GetChoiceIdentifierValues(questionType, testNameIdentifier, itemNumberIdentifier)));
                }
            }
            return (correctChoiceArray, correctAnswerArray);
        }

        private List<(string, string)> GetChoiceIdentifierValues(int questionType, string testNameIdentifier, string itemNumberIdentifier)
        {
            List<(string, string)> choiceIdentifierValueTuple = new List<(string, string)>();

            if (questionType == 7)
            {
                XmlReader xmlReaderInlineChoice = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
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
                XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
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

            return choiceIdentifierValueTuple;
        }

        /*private (double, List<string>, string, string) LoadDeliveryExecutionInfo(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string responseIdentifier, string deliveryExecutionIdentifier,
            List<string> correctAnswerArray, List<string> correctChoiceArray, int subquestionPoints, bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, bool deliveryExecutionFileCreated, int currentSubitemIndex)
        {
            List<string> studentsAnswers = new List<string>();
            string subitemIdentifier = responseIdentifier;
            int amountOfSubitems = GetAmountOfSubitems(testNameIdentifier, itemNumberIdentifier);
            int questionType = GetQuestionType(subitemIdentifier, amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            List<(string, string)> choiceIdentifierValueTuple = GetChoiceIdentifierValues(questionType, testNameIdentifier, itemNumberIdentifier);
            List<string> gapIdentifiers = LoadGapIdentifiers(testNameIdentifier, itemNumberIdentifier);
            double studentsReceivedPoints = 0;
            List<double> studentsReceivedPointsArray = new List<double>();

            foreach (var directory in Directory.GetDirectories(Settings.GetResultsPath()))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string[] attemptIdentifierSplitByUnderscore = Path.GetFileNameWithoutExtension(file).Split("_");
                    if (attemptIdentifierSplitByUnderscore.Length > 2 && attemptIdentifierSplitByUnderscore[2] == deliveryExecutionIdentifier)
                    {
                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "itemResult")
                            {
                                if (xmlReader.GetAttribute("identifier") != itemNameIdentifier)
                                {
                                    xmlReader.Skip();
                                }
                            }

                            if (xmlReader.Name == "responseVariable")
                            {
                                if (xmlReader.GetAttribute("identifier") != subitemIdentifier)
                                {
                                    xmlReader.Skip();
                                }
                            }

                            if (xmlReader.Name == "outcomeVariable")
                            {
                                xmlReader.Skip();
                            }

                            if (xmlReader.Name == "value")
                            {
                                string studentsAnswer = xmlReader.ReadElementContentAsString();
                                if (questionType == 3 || questionType == 4 || questionType == 9)
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
                                else if (questionType == 5 || questionType == 8)
                                {
                                    studentsAnswers.Add(studentsAnswer);
                                }
                                else
                                {
                                    string[] studentsAnswerSplitByApostrophe = studentsAnswer.Split("'");
                                    if (studentsAnswerSplitByApostrophe.Length > 1)
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

            for (int i = 0; i < studentsAnswers.Count; i++)
            {
                if (studentsAnswers[i] == "<>")
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
                for (int j = 0; j < choiceIdentifierValueTuple.Count; j++)
                {
                    if (studentsAnswers[i] == choiceIdentifierValueTuple[j].Item1)
                    {
                        if (questionType == 3 || questionType == 4)
                        {
                            if (answerNumber % 2 == 1)
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + Environment.NewLine;
                            }
                            else
                            {
                                studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + " -> ";
                            }
                            answerNumber++;
                        }
                        else
                        {
                            studentsAnswerToLabel += choiceIdentifierValueTuple[j].Item2 + Environment.NewLine;
                        }
                    }
                }

                if (studentsAnswers[i] != "")
                {
                    studentAnsweredQuestion = true;
                }
            }

            if (questionType == 9)
            {
                studentsAnswerToLabel = "";
                int gapNumber = 0;

                for (int i = 0; i < gapIdentifiers.Count; i++)
                {
                    bool gapAnswered = false;
                    for (int j = 0; j < studentsAnswers.Count; j++)
                    {
                        if (j % 2 == 1)
                        {
                            continue;
                        }
                        if (studentsAnswers[j] == gapIdentifiers[i])
                        {
                            for (int k = 0; k < choiceIdentifierValueTuple.Count; k++)
                            {
                                if (studentsAnswers[j + 1] == choiceIdentifierValueTuple[k].Item1)
                                {
                                    gapAnswered = true;
                                    gapNumber++;
                                    studentsAnswerToLabel += "[" + gapNumber + "] - " + choiceIdentifierValueTuple[k].Item2 + Environment.NewLine;
                                }
                            }
                        }
                    }
                    if (!gapAnswered)
                    {
                        gapNumber++;
                        studentsAnswerToLabel += "[" + gapNumber + "] - nezodpovězeno" + Environment.NewLine;
                    }
                }
            }

            if (questionType == 5 || questionType == 8 || questionType == 10)
            {
                studentsAnswerToLabel = studentsAnswers[0];
            }

            if (!studentAnsweredQuestion)
            {
                studentsAnswerToLabel = "Nevyplněno";
            }

            studentsReceivedPoints = 0;
            StudentsAnswerCorrectness isAnswerCorrect = StudentsAnswerCorrectness.Correct;

            if (deliveryExecutionFileCreated)
            {
                List<double> importedReceivedPointsArray = new List<double>();
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

                if (amountOfSubitems > 1)
                {
                    studentsReceivedPoints = importedReceivedPointsArray[currentSubitemIndex];
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
                    case int n when (n == 1 || n == 6 || n == 7 || n == 10):
                        bool areStudentsAnswersCorrect = Enumerable.SequenceEqual(correctChoiceArray, studentsAnswers);
                        if (areStudentsAnswersCorrect)
                        {
                            studentsReceivedPoints = subquestionPoints;
                        }
                        else
                        {
                            if (studentsAnswers.Count == 0 || (studentsAnswers.Count > 0 && studentsAnswers[0] != ""))
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
                                {
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
                        if (correctAnswerArray[0] == studentsAnswers[0])
                        {
                            studentsReceivedPoints = subquestionPoints;
                        }
                        else
                        {
                            if (studentsAnswers.Count > 0 && studentsAnswers[0] != "")
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
                }
            }

            switch (questionType)
            {
                case int n when (n == 1 || n == 6 || n == 7 || n == 10):
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
                            {
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

            string studentsAnswerCorrectLabel = "";
            switch (isAnswerCorrect)
            {
                case StudentsAnswerCorrectness.Correct:
                    studentsAnswerCorrectLabel = "Správná odpověď.";
                    break;
                case StudentsAnswerCorrectness.PartiallyCorrect:
                    studentsAnswerCorrectLabel = "Částečně správná odpověď.";
                    break;
                case StudentsAnswerCorrectness.Incorrect:
                    studentsAnswerCorrectLabel = "Nesprávná odpověď.";
                    break;
                case StudentsAnswerCorrectness.Unknown:
                    studentsAnswerCorrectLabel = "Otevřená odpověď, body budou přiděleny manuálně.";
                    break;
            }

            if ((studentsReceivedPoints < 0 && !NegativePoints(testNameIdentifier, testNumberIdentifier)) || (studentsAnswers.Count > 0 && studentsAnswers[0] == ""))
            {
                studentsReceivedPoints = 0;
            }

            string studentsAnswerPointsLabel = "Počet bodů za odpověď: " + studentsReceivedPoints + "/" + subquestionPoints;
            /*if (determ)
            {
                studentsAnswerPointsLabel = "Počet bodů za odpověď: N/A";
            }
            else
            {
                studentsAnswerPointsLabel = "Počet bodů za odpověď: " + studentsReceivedPoints + "/" + subquestionPoints;
            }*/
        /*studentsReceivedPointsArray.Add(Math.Round(studentsReceivedPoints, 2));
        List<string> studentsAnswersList = ConvertStudentsAnswersToList(studentsAnswerToLabel);
        return (Math.Round(studentsReceivedPoints, 2), studentsAnswersList, studentsAnswerCorrectLabel, studentsAnswerPointsLabel);
    }*/

        private List<string> LoadGapIdentifiers(string testNameIdentifier, string itemNumberIdentifier)
        {
            List<string> gapIdentifiers = new List<string>();
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "gap")
                {
                    gapIdentifiers.Add(xmlReader.GetAttribute("identifier"));
                }
            }
            return gapIdentifiers;
        }

        private (bool, double, int, bool) LoadQuestionPoints(string testNameIdentifier, string itemNumberIdentifier, string responseIdentifier, int amountOfSubitems, double correctChoicePoints)
        {
            string subitemIdentifier = responseIdentifier;

            bool fileExists = false;
            bool itemRecordExists = false;
            bool questionPointsDetermined = true;
            bool recommendedWrongChoicePoints = false;
            double selectedWrongChoicePoints = 0;
            int questionPoints = 0;
            if (File.Exists(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier)))
            {
                string[] importedFileLines = LoadPoints(testNameIdentifier, itemNumberIdentifier);
                for (int i = 0; i < importedFileLines.Length; i++)
                {
                    string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                    if (splitImportedFileLineBySemicolon[1] == "N/A" || importedFileLines.Length != amountOfSubitems)
                    {
                        questionPointsDetermined = false;
                    }

                    if (splitImportedFileLineBySemicolon[0] == subitemIdentifier)
                    {
                        itemRecordExists = true;

                        if (splitImportedFileLineBySemicolon.Length > 2 && splitImportedFileLineBySemicolon[2] != "N/A")
                        {
                            if (double.Parse(splitImportedFileLineBySemicolon[2]) == correctChoicePoints * (-1))
                            {
                                recommendedWrongChoicePoints = true;
                            }
                            else
                            {
                                recommendedWrongChoicePoints = false;
                                selectedWrongChoicePoints = double.Parse(splitImportedFileLineBySemicolon[2]);
                            }
                        }
                    }

                    if (splitImportedFileLineBySemicolon[1] != "N/A")
                    {
                        questionPoints += int.Parse(splitImportedFileLineBySemicolon[1]);
                    }
                }

                if (!itemRecordExists)
                {
                    string fileLinesToExport = subitemIdentifier + ";N/A" + Environment.NewLine;
                    AppendPoints(testNameIdentifier, itemNumberIdentifier, fileLinesToExport);
                }
            }
            else
            {
                string fileLinesToExport = subitemIdentifier + ";N/A;N/A" + Environment.NewLine;
                SavePoints(testNameIdentifier, itemNumberIdentifier, fileLinesToExport);
            }

            return (recommendedWrongChoicePoints, selectedWrongChoicePoints, questionPoints, questionPointsDetermined);
        }

        private double GetCorrectChoicePoints(int subquestionPoints, List<string> correctChoiceArray, int questionType)
        {
            double correctChoicePoints = 0;
            switch (questionType)
            {
                case int n when (n == 1 || n == 5 || n == 6 || n == 7 || n == 8 || n == 10):
                    correctChoicePoints = subquestionPoints;
                    break;
                case 2:
                    correctChoicePoints = (double)subquestionPoints / (double)correctChoiceArray.Count;
                    break;
                case int n when (n == 3 || n == 4 || n == 9):
                    correctChoicePoints = (double)subquestionPoints / (double)(correctChoiceArray.Count / 2);
                    break;
            }
            /* if (correctChoicePoints == Double.PositiveInfinity || correctChoicePoints == Double.NegativeInfinity)
             {
                 string errorMessage = "Chyba: otázka nemá pravděpodobně zadané žádné správné odpovědi.\nIdentifikátory otázky: " + @ViewBag.itemNameIdentifier + ", " + @ViewBag.itemNumberIdentifier;
                 WriteMessageToUser(errorMessage);
             }*/

            return Math.Round(correctChoicePoints, 2);
        }

        /*private (string, string, string, string, int) LoadItemParameters(string testNameIdentifier, string itemNameIdentifier, string itemNumberIdentifier)
        {
            string title = "";
            string label = "";

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (xmlReader.Name == "assessmentItem")
                    {
                        title = xmlReader.GetAttribute("title");
                        label = xmlReader.GetAttribute("label");
                    }
                }
            }

            int amountOfSubitems = GetAmountOfSubitems(testNameIdentifier, itemNumberIdentifier);

            return (itemNameIdentifier, itemNumberIdentifier, title, label, amountOfSubitems);
        }*/

        private string GetQuestionTypeText(int questionType)
        {
            switch (questionType)
            {
                case 0:
                    return "Neznámý nebo nepodporovaný typ otázky!";
                case 1:
                    return "Seřazení pojmů";
                case 2:
                    return "Výběr z více možností; více možných správných odpovědí";
                case 3:
                    return "Spojování pojmů";
                case 4:
                    return "Více otázek k jednomu pojmu; více možných správných odpovědí";
                case 5:
                    return "Volná odpověď, správná odpověď není automaticky určena";
                case 6:
                    return "Výběr z více možností; jedna správná odpověd";
                case 7:
                    return "Výběr z více možností (doplnění textu); jedna správná odpověď";
                case 8:
                    return "Volná odpověď, správná odpověď je automaticky určena";
                case 9:
                    return "Dosazování pojmů do mezer";
                case 10:
                    return "Posuvník; jedna správná odpověď (číslo)";
                default:
                    return "Neznámý nebo nepodporovaný typ otázky!";
            }
        }

        private double GetStudentsSubitemPoints(List<double> studentsSubitemPointsList, string responseIdentifier, List<string> responseIdentifierArray)
        {
            for (int i = 0; i < responseIdentifierArray.Count; i++)
            {
                if (responseIdentifierArray[i] == responseIdentifier)
                {
                    return studentsSubitemPointsList[i];
                }
            }
            return 0;
        }

        private List<string> ConvertStudentsAnswersToList(string studentsAnswersToLabel)
        {
            List<string> studentsAnswers = new List<string>();
            string[] splitStudentsAnswersByNewLine = studentsAnswersToLabel.Split("\n");
            for (int i = 0; i < splitStudentsAnswersByNewLine.Length; i++)
            {
                studentsAnswers.Add(splitStudentsAnswersByNewLine[i]);
            }
            return studentsAnswers;
        }

        private int GetCurrentSubitemIndex(string responseIdentifier, List<string> responseIdentifierArray)
        {
            for (int i = 0; i < responseIdentifierArray.Count; i++)
            {
                if (responseIdentifierArray[i] == responseIdentifier)
                {
                    return i;
                }
            }
            return 0;
        }

        /*----- Class functions -----*/
        public List<Item> Load(string testNameIdentifier)
        {
            Test test = testData.Load(testNameIdentifier);

            return Load(testNameIdentifier, test);
        }

        public List<Item> Load(string testNameIdentifier, Test test)
        {
            List<Item> items = new List<Item>();

            foreach (TestItem testItem in test.Items)
            {
                string itemNumberIdentifier = testItem.NumberIdentifier;

                Item item = Load(testNameIdentifier, itemNumberIdentifier);
                items.Add(item);
            }

            return items;
        }

        public Item Load(string testNameIdentifier, string itemNumberIdentifier)
        {
            Item item = null;
            ItemResponse itemResponse = null;

            List<string> responseIdentifierArray = new List<string>();
            List<string> responseValueArray = new List<string>();
            string questionText = "";
            int amountOfSubitems = 0;
            int amountOfAddedFaultyQuestions = 0;
            int errorMessageNumber = 0;
            List<(bool, string, string)> includesImage = new List<(bool, string, string)>();

            byte lastRead = 0;

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "assessmentItem")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        item = new Item(
                            xmlReader.GetAttribute("identifier"),
                            xmlReader.GetAttribute("title"),
                            xmlReader.GetAttribute("label"),
                            bool.Parse(xmlReader.GetAttribute("adaptive")),
                            bool.Parse(xmlReader.GetAttribute("timeDependent"))//,
                                                                               //xmlReader.GetAttribute("toolName"),
                            /*xmlReader.GetAttribute("toolVersion")*/,
                            File.Exists(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier)),
                            itemNumberIdentifier);
                        lastRead = 1;
                    }
                }

                if (xmlReader.Name == "responseDeclaration")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        amountOfSubitems++;
                        lastRead = 2;
                    }
                }

                if (xmlReader.GetAttribute("responseIdentifier") != null)
                {
                    string responseIdentifier = xmlReader.GetAttribute("responseIdentifier");
                    responseIdentifierArray.Add(responseIdentifier);
                    int questionType = GetQuestionType(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems);
                    if (questionType == 7 || questionType == 8)
                    {
                        amountOfAddedFaultyQuestions++;
                        string faultyQuestionValue = GetFaultyQuestionValue(testNameIdentifier, itemNumberIdentifier, amountOfAddedFaultyQuestions);
                        responseValueArray.Add(faultyQuestionValue);
                    }

                    (int amountOfImages, questionText, _, includesImage, errorMessageNumber) = SubitemImages(testNameIdentifier, itemNumberIdentifier, responseIdentifier, includesImage);

                    if (responseIdentifierArray.Count - 1 > responseValueArray.Count)
                    {
                        responseValueArray.Add("Otázka nebyla vyplněna.");
                    }
                }

                if (responseIdentifierArray.Count > 0)
                {
                    if (includesImage.Count == 0)
                    {
                        errorMessageNumber = 2;
                    }
                    else
                    {
                        if (xmlReader.Name == "gapMatchInteraction")
                        {
                            using (var innerReader = xmlReader.ReadSubtree())
                            {
                                while (innerReader.Read())
                                {

                                    if (innerReader.Name == "p")
                                    {
                                        string gapText = innerReader.ReadInnerXml();
                                        questionText = "";
                                        bool addText = true;
                                        int gapCounter = 1;

                                        for (int i = 0; i < gapText.Length; i++)
                                        {
                                            if (gapText[i] == '<')
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
                                            if (addText)
                                            {
                                                questionText += gapText[i];
                                            }
                                        }
                                        responseValueArray.Add(questionText);
                                    }
                                }
                            }
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
                                    string responseValue = questionText;
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

                // #GetResponseIdentifiers
            }

            for (int i = 0; i < includesImage.Count; i++)
            {
                if (includesImage[i].Item1 && includesImage[i].Item2 == "")
                {
                    includesImage[i] = (includesImage[i].Item1, responseValueArray[i], includesImage[i].Item3);
                }

                if (includesImage[i].Item1 && includesImage[i].Item2 == "")
                {
                    XmlReader xmlReaderCorrection = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
                    while (xmlReaderCorrection.Read())
                    {
                        if (xmlReaderCorrection.Name == "choiceInteraction")
                        {
                            if (xmlReaderCorrection.GetAttribute("responseIdentifier") != responseIdentifierArray[i])
                            {
                                xmlReaderCorrection.Skip();
                            }
                        }
                        if (xmlReaderCorrection.Name == "gapMatchInteraction")
                        {
                            xmlReaderCorrection.Skip();
                        }
                        if (xmlReaderCorrection.Name == "prompt")
                        {
                            string promptQuestionText = xmlReaderCorrection.ReadInnerXml();
                            int firstStartTag = promptQuestionText.IndexOf('<');
                            int lastEndTag = promptQuestionText.LastIndexOf('>');
                            string questionTextNew = promptQuestionText.Substring(0, firstStartTag) + promptQuestionText.Substring(1 + lastEndTag);
                            responseValueArray[i] = questionTextNew;
                            includesImage[i] = (includesImage[i].Item1, responseValueArray[i], includesImage[i].Item3);
                            break;
                        }
                    }
                }
            }

            // Responses
            item.Responses = new List<ItemResponse>();
            ItemResponse response;
            if (File.Exists(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier)))
            {
                string[] importedFileLines = LoadPoints(testNameIdentifier, itemNumberIdentifier);

                for (int i = 0; i < importedFileLines.Length; i++)
                {
                    // Points
                    string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                    string responseIdentifier = splitImportedFileLineBySemicolon[0];
                    string correctChoicePoints = splitImportedFileLineBySemicolon[1];
                    if (splitImportedFileLineBySemicolon.Length > 2)
                    {
                        string wrongChoicePoints = splitImportedFileLineBySemicolon[2];
                        response = new ItemResponse(responseIdentifier, correctChoicePoints, wrongChoicePoints);
                    }
                    else
                    {
                        response = new ItemResponse(responseIdentifier, correctChoicePoints);
                    }
                    item.Responses.Add(response);
                }
            }
            else
            {
                response = new ItemResponse(responseIdentifierArray[0], "N/A", "N/A");
                item.Responses.Add(response);
            }

            return item;
        }

        //-------------------------------------------------- #OLD --------------------------------------------------

        /*private void SavePoints(string testNameIdentifier, string itemNumberIdentifier, string points)
        {
            string file = Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier);
            File.WriteAllText(file, points);
        }*/

        /*public List<Item> Load(string testNameIdentifier)
        {
            List<Item> items = new List<Item>();

            foreach (var directory in Directory.GetDirectories(Settings.GetTestItemsPath(testNameIdentifier)))
            {
                string itemNumberIdentifier = Path.GetFileName(directory);

                Item item = Load(testNameIdentifier, itemNumberIdentifier);
                items.Add(item);
            }

            return items;
        }

        public Item Load(string testNameIdentifier, string itemNumberIdentifier)
        {
            Item item = null;

            byte lastRead = 0;

            // Load XML
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "assessmentItem")
                {
                    if (xmlReader.NodeType != XmlNodeType.EndElement)
                    {
                        item = new Item(
                            xmlReader.GetAttribute("identifier"),
                            xmlReader.GetAttribute("title"),
                            xmlReader.GetAttribute("label"),
                            bool.Parse(xmlReader.GetAttribute("adaptive")),
                            bool.Parse(xmlReader.GetAttribute("timeDependent")),
                            xmlReader.GetAttribute("toolName"),
                            xmlReader.GetAttribute("toolVersion"));
                        lastRead = 1;
                    }
                }
            }
            if (xmlReader.Name == "assessmentTest")
            {
                if (xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    test = new Test(
                        xmlReader.GetAttribute("identifier"),
                        xmlReader.GetAttribute("title"),
                        xmlReader.GetAttribute("toolName"),
                        xmlReader.GetAttribute("toolVersion"));
                    lastRead = 1;
                }
            }

            if (xmlReader.Name == "testPart")
            {
                if (xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    part = new TestPart(
                        xmlReader.GetAttribute("identifier"),
                        xmlReader.GetAttribute("navigationMode"),
                        xmlReader.GetAttribute("submissionMode"));
                    lastRead = 2;
                }
                else if (test != null)
                {
                    test.Parts.Add(part);
                }
            }

            if (xmlReader.Name == "assessmentSection")
            {
                if (xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    section = new TestSection(
                        xmlReader.GetAttribute("identifier"),
                        bool.Parse(xmlReader.GetAttribute("required")),
                        bool.Parse(xmlReader.GetAttribute("fixed")),
                        xmlReader.GetAttribute("title"),
                        bool.Parse(xmlReader.GetAttribute("visible")),
                        bool.Parse(xmlReader.GetAttribute("keepTogether")));
                    lastRead = 3;
                }
                else if (part != null)
                {
                    part.Sections.Add(section);
                }
            }

            if (xmlReader.Name == "assessmentItemRef")
            {
                if (xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    item = new TestItem(
                        xmlReader.GetAttribute("identifier"),
                        bool.Parse(xmlReader.GetAttribute("required")),
                        bool.Parse(xmlReader.GetAttribute("fixed")),
                        xmlReader.GetAttribute("href"));
                    lastRead = 4;
                }
                else if (section != null)
                {
                    section.Items.Add(item);
                }
            }

            if (xmlReader.Name == "itemSessionControl")
            {
                SessionControl sessionControl = new SessionControl(
                    int.Parse(xmlReader.GetAttribute("maxAttempts")),
                    bool.Parse(xmlReader.GetAttribute("showFeedback")),
                    bool.Parse(xmlReader.GetAttribute("allowReview")),
                    bool.Parse(xmlReader.GetAttribute("showSolution")),
                    bool.Parse(xmlReader.GetAttribute("allowComment")),
                    bool.Parse(xmlReader.GetAttribute("allowSkipping")),
                    bool.Parse(xmlReader.GetAttribute("validateResponses")));

                switch (lastRead)
                {
                    case 1:
                        test.SessionControl = sessionControl;
                        break;
                    case 2:
                        part.SessionControl = sessionControl;
                        break;
                    case 3:
                        section.SessionControl = sessionControl;
                        break;
                    case 4:
                        item.SessionControl = sessionControl;
                        break;
                }
                lastRead = 5;
            }

            if (xmlReader.Name == "timeLimits")
            {
                TimeLimits timeLimits = new TimeLimits(
                    bool.Parse(xmlReader.GetAttribute("allowLateSubmission")));

                switch (lastRead)
                {
                    case 1:
                        test.TimeLimits = timeLimits;
                        break;
                    case 2:
                        part.TimeLimits = timeLimits;
                        break;
                    case 3:
                        section.TimeLimits = timeLimits;
                        break;
                    case 4:
                        item.TimeLimits = timeLimits;
                        break;
                }
                lastRead = 6;
            }
        }

        // Points
        for (int i = 0; i < test.Parts.Count; i++)
        {
            for (int j = 0; j < test.Parts[i].Sections.Count; j++)
            {
                for (int k = 0; k < test.Parts[i].Sections[j].Items.Count; k++)
                {
                    foreach (var file in Directory.GetFiles(Settings.Path + "\\tests\\" + testNameIdentifier + "\\items\\" + test.Items[k].Href.Split("/")[3]))
                    {
                        if (Path.GetFileName(file) == "Points.txt")
                        {
                            test.Parts[i].Sections[j].Items[k].PointsDetermined = true;

                            string[] importedFileLines = File.ReadAllLines(file);
                            for (int l = 0; l < importedFileLines.Length; l++)
                            {
                                string[] splitImportedFileLineBySemicolon = importedFileLines[l].Split(";");

                                if (splitImportedFileLineBySemicolon[1] == "N/A")
                                {
                                    test.Parts[i].Sections[j].Items[k].PointsDetermined = false;
                                }
                                else
                                {
                                    test.Parts[i].Sections[j].Items[k].Points += int.Parse(splitImportedFileLineBySemicolon[1]);
                                }
                            }

                            break;
                        }
                    }

                    if (!test.Parts[i].Sections[j].Items[k].PointsDetermined)
                    {
                        test.PointsDetermined = false;
                    }
                }
            }
        }

        // Negative points
        foreach (var file in Directory.GetFiles(Settings.Path + "\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier))
        {
            if (Path.GetFileName(file) == "NegativePoints.txt")
            {
                string[] negativePointsFileLines = File.ReadAllLines(file);
                if (negativePointsFileLines[0] == "1")
                {
                    test.NegativePoints = true;
                }
            }
        }

        return test;

            return item;
        }*/
    }
}
