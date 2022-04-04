using System.Diagnostics;
using System.Xml;
using Common;

namespace ViewLayer.Controllers
{
    public class ItemController
    {
        public int CreateNewResultPointsFile(string testNameIdentifier, string testNumberIdentifier, string deliveryExecutionIdentifier, List<(string, string, string, string, int, bool)> itemParameters)
        {
            string resultPointsToText = "";
            int errorMessageNumber = 0;

            for (int i = 0; i < itemParameters.Count; i++)
            {
                string itemNumberIdentifier = itemParameters[i].Item1;
                string itemNameIdentifier = itemParameters[i].Item2;
                foreach (var directory in Directory.GetDirectories(Settings.GetTestItemsPath(testNameIdentifier)))
                {
                    string[] directoryName = directory.Split(@"\");
                    if (itemNumberIdentifier == directoryName[directoryName.Length - 1])
                    {
                        resultPointsToText += itemNameIdentifier;
                        int amountOfSubitems = GetAmountOfSubitems(testNameIdentifier, itemNumberIdentifier);
                        (List<string> responseIdentifierArray, List<string> responseValueArray, errorMessageNumber) = GetResponseIdentifiers(amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
                        foreach (string responseIdentifier in responseIdentifierArray)
                        {
                            string responseIdentifierTemp = responseIdentifier;
                            string imageSource = "", subitemText = "";
                            bool subquestionPointsDetermined;
                            List<string> possibleAnswerArray = new List<string>();
                            List<string> subquestionArray = new List<string>();
                            List<string> correctChoiceArray = new List<string>();
                            List<string> correctAnswerArray = new List<string>();
                            int questionType = 0, subquestionPoints = 0;
                            double wrongChoicePoints = 0;
                            (responseIdentifierTemp, questionType, subquestionPoints, subquestionPointsDetermined, wrongChoicePoints, imageSource, subitemText, possibleAnswerArray, subquestionArray, correctChoiceArray, correctAnswerArray) = LoadSubitemParameters(responseIdentifier, amountOfSubitems, responseIdentifierArray, responseValueArray, testNameIdentifier, itemNumberIdentifier);
                            double correctChoicePoints = GetCorrectChoicePoints(subquestionPoints, correctChoiceArray, questionType);
                            (bool recommendedWrongChoicePoints, double selectedWrongChoicePoints, int questionPoints, bool questionPointsDetermined) = LoadQuestionPoints(testNameIdentifier, itemNumberIdentifier, responseIdentifier, amountOfSubitems, correctChoicePoints);
                            (double studentsSubitemPoints, _, _, _) = LoadDeliveryExecutionInfo(testNameIdentifier, testNumberIdentifier, itemNumberIdentifier, itemNameIdentifier, responseIdentifier, deliveryExecutionIdentifier, correctAnswerArray, correctChoiceArray, subquestionPoints, recommendedWrongChoicePoints, selectedWrongChoicePoints, false, GetCurrentSubitemIndex(responseIdentifier, responseIdentifierArray));
                            { }
                            resultPointsToText += ";" + studentsSubitemPoints.ToString();
                        }
                        resultPointsToText += "\n";
                    }
                }
            }
            File.WriteAllText(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier), resultPointsToText);
            return errorMessageNumber;
        }

        public int GetAmountOfSubitems(string testNameIdentifier, string itemNumberIdentifier)
        {
            int amountOfSubitems = 0;
            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "responseDeclaration" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    amountOfSubitems++;
                }
            }
            return amountOfSubitems;
        }

        public (List<string>, List<string>, int) GetResponseIdentifiers(int amountOfSubitems, string testNameIdentifier, string itemNumberIdentifier)
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
                    int questionType = GetQuestionType(responseIdentifier, amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
                    if (questionType == 7 || questionType == 8)
                    {
                        amountOfAddedFaultyQuestions++;
                        string faultyQuestionValue = GetFaultyQuestionValue(amountOfAddedFaultyQuestions, testNameIdentifier, itemNumberIdentifier);
                        List<int> amountOfInlineChoiceInteractions = GetAmountOfInlineChoiceInteractions(testNameIdentifier, itemNumberIdentifier);
                        for (int i = 0; i < amountOfInlineChoiceInteractions.Count; i++)
                        {
                            int amountOfSubitemInlineChoiceInterractions = amountOfInlineChoiceInteractions[i];
                            if (amountOfSubitemInlineChoiceInterractions > 1)
                            {
                                faultyQuestionValue = "Při přidávání otázky nastala chyba, jedna podotázka nemůže obsahovat více než 1 doplnění pojmu do textu. Chybu prosím opravte v nástroji TAO Core.";
                            }
                            continue;
                        }
                        responseValueArray.Add(faultyQuestionValue);
                    }

                    (int amountOfImages, questionText, _, includesImage, errorMessageNumber) = SubitemImages(responseIdentifier, includesImage, testNameIdentifier, itemNumberIdentifier);

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

            if (includesImage.Count != responseValueArray.Count)
            {
                responseValueArray.Clear();
                for (int i = 0; i < includesImage.Count; i++)
                {
                    responseValueArray.Add("Chyba - některé z otázek nemají zadaný text. Texty podotázek nebudou pro tuto otázku fungovat.");
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
                        if (xmlReaderCorrection.GetAttribute("responseIdentifier") != null)
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

        public int GetQuestionType(string responseIdentifierCorrection, int amountOfSubitems, string testNameIdentifier, string itemNumberIdentifier)
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
                        questionType = 10;//Typ otázky = volná odpověď; odpověď není předem daná
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

        public string GetFaultyQuestionValue(int amountOfAddedFaultyQuestions, string testNameIdentifier, string itemNumberIdentifier)
        {
            int amountOfFaultyQuestions = 0;

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "gapMatchInteraction")
                {
                    xmlReader.Skip();
                }

                if (xmlReader.Name == "div" && xmlReader.GetAttribute("class") == "col-12" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    using (var innerReader = xmlReader.ReadSubtree())
                    {
                        while (innerReader.Read())
                        {
                            if (innerReader.Name == "p")
                            {
                                string inlineChoiceInteractionLine_ = innerReader.ReadInnerXml();
                                if (inlineChoiceInteractionLine_.Length > 0)
                                {
                                    if (inlineChoiceInteractionLine_[0] != '<' && inlineChoiceInteractionLine_.Substring(0, 1) != "\n")
                                    {
                                        amountOfFaultyQuestions++;
                                        if (amountOfAddedFaultyQuestions != amountOfFaultyQuestions)
                                        {
                                            innerReader.Skip();
                                        }
                                        else
                                        {
                                            int firstStartTag = inlineChoiceInteractionLine_.IndexOf('<');
                                            int lastEndTag = inlineChoiceInteractionLine_.LastIndexOf('>');
                                            string questionText = inlineChoiceInteractionLine_.Substring(0, firstStartTag) + "(DOPLŇTE)" + inlineChoiceInteractionLine_.Substring(1 + lastEndTag);
                                            return questionText;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "gapMatchInteraction")
                {
                    xmlReader.Skip();
                }

                if (xmlReader.Name == "div" && xmlReader.GetAttribute("class") == "col-12" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    using (var innerReader = xmlReader.ReadSubtree())
                    {
                        while (innerReader.Read())
                        {
                            string inlineChoiceInteractionLine_ = innerReader.ReadInnerXml();
                            if (inlineChoiceInteractionLine_.Length > 0)
                            {
                                if (inlineChoiceInteractionLine_[0] != '<' && inlineChoiceInteractionLine_.Substring(0, 1) != "\n")
                                {
                                    amountOfFaultyQuestions++;
                                    if (amountOfAddedFaultyQuestions != amountOfFaultyQuestions)
                                    {
                                        innerReader.Skip();
                                    }
                                    else
                                    {
                                        int firstStartTag = inlineChoiceInteractionLine_.IndexOf('<');
                                        int lastEndTag = inlineChoiceInteractionLine_.LastIndexOf('>');
                                        string questionText = inlineChoiceInteractionLine_.Substring(0, firstStartTag) + "(DOPLŇTE)" + inlineChoiceInteractionLine_.Substring(1 + lastEndTag);
                                        return questionText;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return "Při přidávání otázky nastala neočekávaná chyba";
        }

        public List<int> GetAmountOfInlineChoiceInteractions(string testNameIdentifier, string itemNumberIdentifier)
        {
            List<int> amountOfInlineChoiceInteractions = new List<int>();

            XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
            while (xmlReader.Read())
            {
                if (xmlReader.Name == "p" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    int amountOfSubitemInlineChoiceInterractions = 0;
                    using (var innerReader = xmlReader.ReadSubtree())
                    {
                        while (innerReader.Read())
                        {
                            if ((innerReader.Name == "inlineChoiceInteraction" || innerReader.Name == "textEntryInteraction") && innerReader.NodeType != XmlNodeType.EndElement)
                            {
                                amountOfSubitemInlineChoiceInterractions++;
                            }
                        }
                    }
                    amountOfInlineChoiceInteractions.Add(amountOfSubitemInlineChoiceInterractions);
                }
            }

            return amountOfInlineChoiceInteractions;
        }

        public (int, string, string, List<(bool, string, string)>, int) SubitemImages(string responseIdentifier, List<(bool, string, string)> includesImage, string testNameIdentifier, string itemNumberIdentifier)
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
                    if (name == "choiceInteraction" || name == "sliderInteraction" || name == "gapMatchInteraction" || name == "matchInteraction" || name == "extendedTextInteraction"
                        || name == "orderInteraction" || name == "associateInteraction")//chybi volne odpovedi obe?
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

        public (string, int, int, bool, double, string, string, List<string>, List<string>, List<string>, List<string>) LoadSubitemParameters(string responseIdentifier, int amountOfSubitems, List<string> responseIdentifierArray, List<string> responseValueArray, string testNameIdentifier, string itemNumberIdentifier)//načte parametry dané podotázky
        {
            int questionType = GetQuestionType(responseIdentifier, amountOfSubitems, testNameIdentifier, itemNumberIdentifier);
            (int subquestionPoints, bool subquestionPointsDetermined, double wrongChoicePoints) = GetSubquestionPoints(responseIdentifier, amountOfSubitems, questionType, testNameIdentifier, itemNumberIdentifier);
            List<(bool, string, string)> includesImage = new List<(bool, string, string)>();
            string imageSource = "";
            if (SubitemImages(responseIdentifier, includesImage, testNameIdentifier, itemNumberIdentifier).Item3 != "")
            {
                imageSource = Settings.GetImageURL() + testNameIdentifier + "/items/" + itemNumberIdentifier + "/" + SubitemImages(responseIdentifier, includesImage, testNameIdentifier, itemNumberIdentifier).Item3;
            }
            string subitemText = "";

            if (responseIdentifierArray.Count != responseValueArray.Count)
            {
                responseValueArray.Clear();
                for (int i = 0; i < responseIdentifierArray.Count; i++)
                {
                    responseValueArray.Add("Chyba - některé z otázek nemají zadaný text. Texty podotázek nebudou pro tuto otázku fungovat.");
                }
            }

            for (int i = 0; i < responseIdentifierArray.Count; i++)
            {
                if (responseIdentifier == responseIdentifierArray[i])
                {
                    subitemText = responseValueArray[i];
                }
            }

            (List<string> possibleAnswerArray, List<string> subquestionArray) = GetPossibleAnswerList(responseIdentifier, amountOfSubitems, questionType, testNameIdentifier, itemNumberIdentifier);
            (List<string> correctChoiceArray, List<string> correctAnswerArray) = GetCorrectAnswerList(responseIdentifier, amountOfSubitems, questionType, testNameIdentifier, itemNumberIdentifier);

            return (responseIdentifier, questionType, subquestionPoints, subquestionPointsDetermined, wrongChoicePoints, imageSource, subitemText, possibleAnswerArray, subquestionArray, correctChoiceArray, correctAnswerArray);
        }

        public (int, bool, double) GetSubquestionPoints(string responseIdentifier, int amountOfSubitems, int questionType, string testNameIdentifier, string itemNumberIdentifier)
        {
            bool fileExists = false;
            int subquestionPoints = 0;
            bool subquestionPointsDetermined = true;
            double wrongChoicePoints = 0;

            if (File.Exists(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier)))
            {
                fileExists = true;
            }

            if (!fileExists)
            {
                subquestionPointsDetermined = false;
                string itemPointsText = responseIdentifier + ";N/A;N/A" + Environment.NewLine;
                File.WriteAllText(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier), itemPointsText);
            }
            else
            {
                string[] importedFileLines = File.ReadAllLines(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier));
                for (int i = 0; i < importedFileLines.Length; i++)
                {
                    string[] splitImportedFileLineBySemicolon = importedFileLines[i].Split(";");
                    if (splitImportedFileLineBySemicolon[1] == "N/A" || importedFileLines.Length != amountOfSubitems)
                    {
                        subquestionPointsDetermined = false;
                    }

                    if (splitImportedFileLineBySemicolon[0] == responseIdentifier)
                    {
                        if (splitImportedFileLineBySemicolon[1] != "N/A")
                        {
                            subquestionPoints = int.Parse(splitImportedFileLineBySemicolon[1]);
                        }

                        if (splitImportedFileLineBySemicolon.Length > 2 && splitImportedFileLineBySemicolon[2] != "N/A")
                        {
                            wrongChoicePoints = double.Parse(splitImportedFileLineBySemicolon[2]);
                        }
                    }
                }
            }

            return (subquestionPoints, subquestionPointsDetermined, wrongChoicePoints);
        }

        public (List<string>, List<string>) GetPossibleAnswerList(string selectedResponseIdentifier, int amountOfSubitems, int questionType, string testNameIdentifier, string itemNumberIdentifier)
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

        public (List<string>, List<string>) GetCorrectAnswerList(string selectedResponseIdentifier, int amountOfSubitems, int questionType, string testNameIdentifier, string itemNumberIdentifier)
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

        public string GetChoiceValue(string choiceIdentifier, List<(string, string)> choiceIdentifierValueTuple)
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

        public List<(string, string)> GetChoiceIdentifierValues(int questionType, string testNameIdentifier, string itemNumberIdentifier)
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

        public (double, List<string>, string, string) LoadDeliveryExecutionInfo(string testNameIdentifier, string testNumberIdentifier, string itemNumberIdentifier, string itemNameIdentifier, string responseIdentifier, string deliveryExecutionIdentifier,
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

            // Debug řádek pokud nesedí body
            studentsAnswers.RemoveRange(studentsAnswers.Count / 2, studentsAnswers.Count / 2);

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
                string[] resultsFileLines = File.ReadAllLines(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier));
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
            /*if (subquestionPoints != 0)
            {*/
            switch (isAnswerCorrect)
            {
                case StudentsAnswerCorrectness.Correct:
                    studentsAnswerCorrectLabel = "Správná odpověď";
                    break;
                case StudentsAnswerCorrectness.PartiallyCorrect:
                    studentsAnswerCorrectLabel = "Částečně správná odpověď";
                    break;
                case StudentsAnswerCorrectness.Incorrect:
                    studentsAnswerCorrectLabel = "Nesprávná odpověď";
                    break;
                case StudentsAnswerCorrectness.Unknown:
                    studentsAnswerCorrectLabel = "Otevřená odpověď, body budou přiděleny manuálně";
                    break;
            }
            //}

            if ((studentsReceivedPoints < 0 && !NegativePoints(testNameIdentifier, testNumberIdentifier)) || (studentsAnswers.Count > 0 && studentsAnswers[0] == ""))
            {
                studentsReceivedPoints = 0;
            }

            string studentsAnswerPointsLabel = "Počet bodů za odpověď: " + studentsReceivedPoints + " / " + subquestionPoints;
            /*if (determ)
            {
                studentsAnswerPointsLabel = "Počet bodů za odpověď: N/A";
            }
            else
            {
                studentsAnswerPointsLabel = "Počet bodů za odpověď: " + studentsReceivedPoints + "/" + subquestionPoints;
            }*/
            studentsReceivedPointsArray.Add(Math.Round(studentsReceivedPoints, 2));
            List<string> studentsAnswersList = ConvertStudentsAnswersToList(studentsAnswerToLabel);
            return (Math.Round(studentsReceivedPoints, 2), studentsAnswersList, studentsAnswerCorrectLabel, studentsAnswerPointsLabel);
        }

        enum StudentsAnswerCorrectness
        {
            Correct,
            Incorrect,
            PartiallyCorrect,
            Unknown
        }

        public List<string> LoadGapIdentifiers(string testNameIdentifier, string itemNumberIdentifier)
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

        public (bool, double, int, bool) LoadQuestionPoints(string testNameIdentifier, string itemNumberIdentifier, string responseIdentifier, int amountOfSubitems, double correctChoicePoints)
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
                fileExists = true;
            }

            if (!fileExists)
            {
                string itemPointsText = subitemIdentifier + ";N/A;N/A" + Environment.NewLine;
                File.WriteAllText(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier), itemPointsText);
            }
            else
            {
                string[] importedFileLines = File.ReadAllLines(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier));
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
                    string itemPointsText = subitemIdentifier + ";N/A" + Environment.NewLine;
                    File.AppendAllText(Settings.GetTestItemPointsDataPath(testNameIdentifier, itemNumberIdentifier), itemPointsText);
                }
            }

            return (recommendedWrongChoicePoints, selectedWrongChoicePoints, questionPoints, questionPointsDetermined);
        }

        public double GetCorrectChoicePoints(int subquestionPoints, List<string> correctChoiceArray, int questionType)
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
            return Math.Round(correctChoicePoints, 2);
        }

        public (string, string, string, string, int) LoadItemParameters(string testNameIdentifier, string itemNameIdentifier, string itemNumberIdentifier)
        {
            string title = "";
            string label = "";
            int amountOfSubitems = 0;

            if (Directory.Exists(Settings.GetTestPath(testNameIdentifier)))
            {
                if (Directory.Exists(Settings.GetTestItemsPath(testNameIdentifier)))
                {
                    if (Directory.Exists(Settings.GetTestItemPath(testNameIdentifier, itemNumberIdentifier)))
                    {
                        if (File.Exists(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier)))
                        {
                            try
                            {
                                XmlReader xmlReader = XmlReader.Create(Settings.GetTestItemFilePath(testNameIdentifier, itemNumberIdentifier));
                                while (xmlReader.Read())
                                {
                                    if (xmlReader.Name == "assessmentItem" && xmlReader.NodeType == XmlNodeType.Element)
                                    {
                                        bool allAttributesFound = true;

                                        if (xmlReader.GetAttribute("title") != null)
                                        {
                                            title = xmlReader.GetAttribute("title");
                                        }
                                        else
                                        {
                                            allAttributesFound = false;
                                        }

                                        if (xmlReader.GetAttribute("label") != null)
                                        {
                                            label = xmlReader.GetAttribute("label");
                                        }
                                        else
                                        {
                                            allAttributesFound = false;
                                        }

                                        if (!allAttributesFound)
                                        {
                                            throw Exceptions.XmlAttributeNotFound;
                                        }
                                    }

                                    if (xmlReader.Name == "responseDeclaration" && xmlReader.NodeType == XmlNodeType.Element)
                                    {
                                        amountOfSubitems++;
                                    }
                                }
                            }
                            catch (XmlException e) when (e.Message == "Root element is missing.")
                            {
                                throw Exceptions.XmlRootElementMissing;
                            }
                        }
                        else { throw Exceptions.TestItemFilePathNotFoundException; }
                    }
                    else { throw Exceptions.TestItemPathNotFoundException; }
                }
                else { throw Exceptions.TestItemsPathNotFoundException; }
            }
            else { throw Exceptions.TestPathNotFoundException; }

            return (itemNameIdentifier, itemNumberIdentifier, title, label, amountOfSubitems);
        }

        public string GetQuestionTypeText(int questionType)
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

        public List<double> GetStudentsSubitemPointsList(string testNameIdentifier, string itemNameIdentifier, string deliveryExecutionIdentifier)
        {
            List<double> studentsSubitemPoints = new List<double>();
            string[] resultsFileLines = File.ReadAllLines(Settings.GetResultResultsDataPath(testNameIdentifier, deliveryExecutionIdentifier));
            for (int i = 0; i < resultsFileLines.Length; i++)
            {
                string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                if (splitResultsFileLineBySemicolon[0] == itemNameIdentifier)
                {
                    for (int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                    {
                        studentsSubitemPoints.Add(double.Parse(splitResultsFileLineBySemicolon[j]));
                    }
                }
            }
            return studentsSubitemPoints;
        }

        public double GetStudentsSubitemPoints(List<double> studentsSubitemPointsList, string responseIdentifier, List<string> responseIdentifierArray)
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

        public List<string> ConvertStudentsAnswersToList(string studentsAnswersToLabel)
        {
            List<string> studentsAnswers = new List<string>();
            string[] splitStudentsAnswersByNewLine = studentsAnswersToLabel.Split("\n");
            for (int i = 0; i < splitStudentsAnswersByNewLine.Length; i++)
            {
                if (splitStudentsAnswersByNewLine[i] != "")
                {
                    studentsAnswers.Add(splitStudentsAnswersByNewLine[i]);
                }
            }
            return studentsAnswers;
        }

        public int GetCurrentSubitemIndex(string responseIdentifier, List<string> responseIdentifierArray)
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

        public bool NegativePoints(string testNameIdentifier, string testNumberIdentifier)
        {
            bool negativePoints = false;
            string testPath = Settings.GetTestTestPath(testNameIdentifier, testNumberIdentifier);
            foreach (var file in Directory.GetFiles(testPath))
            {
                if (Path.GetFileName(file) == "NegativePoints.txt")
                {
                    string[] negativePointsFileLines = File.ReadAllLines(file);
                    for (int i = 0; i < negativePointsFileLines.Length; i++)
                    {
                        if (negativePointsFileLines[0] == "1")
                        {
                            negativePoints = true;
                        }
                    }
                }
            }
            return negativePoints;
        }
    }
}
