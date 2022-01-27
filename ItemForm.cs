using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;

namespace TAO_Enhancer
{
    /*
     * TODO: Sekce (k jaké sekci otázka patří)
     * Obrázky u podotázek
     */
    public partial class ItemForm : Form
    {
        string identifier = "";
        string testNameIdentifier = "";
        string testNumberIdentifier = "";
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

        public ItemForm(string testNameID, string itemNumberID, string testNumberID)
        {
            InitializeComponent();
            testNameIdentifier = testNameID;
            identifier = itemNumberID;
            testNumberIdentifier = testNumberID;
            LoadItemInfo();
        }

        public void LoadItemInfo()
        {
            IdentifierLabel.Text = "Identifikátor: " + identifier;
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

            if (questionType == 3 || questionType == 4)
            {
                GetChoiceIdentifierValues();
            }

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
                        QuestionImage.ImageLocation = ("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + identifier + "\\" + xmlReader.GetAttribute("src"));
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

                if(!includesQuestion)
                {
                    QuestionLabel.Text = "Otázka nebyla vyplněna.";
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
        }

        public string GetItemPath()
        {
            return "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + identifier + "\\qti.xml";
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
            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            bool imageFound = false;
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
                    {
                        questionType = 3;//Typ otázky = spojování párů; TODO 10: Problém s tím, když je jedna možnost ve více párech
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "multiple" && xmlReader.GetAttribute("baseType") == "directedPair")
                    {
                        questionType = 4;//Typ otázky = více otázek (tabulka); více odpovědí může být správně TODO: co když je jen jedno správně?
                    }
                    else if (xmlReader.GetAttribute("cardinality") == "single" && xmlReader.GetAttribute("baseType") == "string")
                    {
                        questionType = 5;//Typ otázky = volná odpověď; TODO 11: Přidat typ otázky, kde je volná otázka ale daná odpověď
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

        private void button1_Click(object sender, EventArgs e)
        {
            new TestForm((testNameIdentifier, testNumberIdentifier)).Show();
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
                ResponseIdentifierLabel.Text = "Identifikátor pododpovědi: " + responseIdentifierArray[SubitemCB.SelectedIndex];
            }
        }
    }
}
