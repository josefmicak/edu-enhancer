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
using VDS.RDF;
using VDS.RDF.Parsing;

namespace TAO_Enhancer
{
    public partial class TestForm : Form
    {
        string testNameIdentifier = "";
        string testNumberIdentifier = "";
        int chosenItem = -1;
        string itemNameIdentifier = "";
        string itemNumberIdentifier = "";
        bool isTeacherEditingQuestion = true;
        string deliveryExecutionIdentifier = "";
        string studentIdentifier = "";
        int testPoints = 0;
        bool testPointsDetermined = true;
        bool isTeacherReviewingDeliveryResult = false;

        public TestForm((string, string) id, bool requestOrigin, string attemptIdentifier, string studentID, bool isTeacherEditingDeliveryResult)
        {
            InitializeComponent();
            testNameIdentifier = id.Item1;
            testNumberIdentifier = id.Item2;
            isTeacherReviewingDeliveryResult = isTeacherEditingDeliveryResult;
            if (!requestOrigin)
            {
                isTeacherEditingQuestion = false;
                deliveryExecutionIdentifier = attemptIdentifier;
                studentIdentifier = studentID;
            }
            else
            {
                resultGB.Visible = false;
            }
            LoadTestInfo();
            if(!requestOrigin)
            {
                LoadResultInfo();
            }
        }

        public void LoadTestInfo()
        {
            TestFolderNameLabel.Text = "Jméno složky: " + testNameIdentifier;
            TestNumberIdentifierLabel.Text = "Číselný identifikátor testu: " + testNumberIdentifier;

            string testitemIdentifier;
            string itemIdentifier;
            string testPart = "";
            string testSection = "";
            int amountOfItems = 0;

            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests\\" + testNumberIdentifier + "\\test.xml");
            while (xmlReader.Read())
            {
                if ((xmlReader.NodeType == XmlNodeType.Element) && (xmlReader.Name == "assessmentTest"))
                {
                    if (xmlReader.HasAttributes)
                    {
                        TestTitleLabel.Text = "Nadpis testu: " + xmlReader.GetAttribute("title");
                        TestNameIdentifierLabel.Text = "Jmenný identifikátor testu: " + xmlReader.GetAttribute("identifier");
                    }
                }

                if(xmlReader.Name == "testPart")
                {
                    testPart = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentSection")
                {
                    testSection = xmlReader.GetAttribute("identifier");
                }

                if (xmlReader.Name == "assessmentItemRef" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    testitemIdentifier = xmlReader.GetAttribute("identifier");
                    itemIdentifier = xmlReader.GetAttribute("href");
                    string[] itemIdentifierSplit = itemIdentifier.Split("/");
                    ItemsGridView.Rows.Add();
                    ItemsGridView.Rows[amountOfItems].Cells[0].Value = testPart;
                    ItemsGridView.Rows[amountOfItems].Cells[1].Value = testSection;
                    ItemsGridView.Rows[amountOfItems].Cells[2].Value = testitemIdentifier;
                    ItemsGridView.Rows[amountOfItems].Cells[3].Value = itemIdentifierSplit[3];
                    amountOfItems++;
                }
            }
            AmountOfQuestionsLabel.Text = "Počet otázek: " + amountOfItems;

            for(int i = 0; i < ItemsGridView.Rows.Count; i++)
            {
                bool questionPointsDetermined = false;
                int questionPoints = 0;
                string itemParentPath = "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + ItemsGridView.Rows[i].Cells[3].Value;
             
                foreach (var file in Directory.GetFiles(itemParentPath))
                {
                    if (Path.GetFileName(file) == "Points.txt")
                    {
                        questionPointsDetermined = true;
                    }
                }

                if(questionPointsDetermined)
                {
                    string[] importedFileLines = File.ReadAllLines(itemParentPath + "\\Points.txt");
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
                }

                if(!questionPointsDetermined)
                {
                    testPointsDetermined = false;
                    ItemsGridView.Rows[i].Cells[4].Value = "N/A";
                }
                else
                {
                    testPoints += questionPoints;
                    ItemsGridView.Rows[i].Cells[4].Value = questionPoints.ToString();
                }
            }
        }

        public void LoadItemInfo()
        {
            ItemNameIdentifierLabel.Text = "Jmenný identifikátor otázky: " + ItemsGridView.Rows[chosenItem].Cells[2].Value.ToString();
            ItemNumberIdentifierLabel.Text = "Číselný identifikátor otázky: " + ItemsGridView.Rows[chosenItem].Cells[3].Value.ToString();
            itemNameIdentifier = ItemsGridView.Rows[chosenItem].Cells[2].Value.ToString();
            itemNumberIdentifier = ItemsGridView.Rows[chosenItem].Cells[3].Value.ToString();
            ItemPointsLabel.Text = "Počet bodů za otázku: " + ItemsGridView.Rows[chosenItem].Cells[4].Value.ToString();

            XmlReader xmlReader = XmlReader.Create(GetItemPath());
            while (xmlReader.Read())
            {
                if (xmlReader.NodeType == XmlNodeType.Element && xmlReader.HasAttributes)
                {
                    if (xmlReader.Name == "assessmentItem")
                    {
                        ItemTitleLabel.Text = "Nadpis otázky: " + xmlReader.GetAttribute("title");
                        ItemLabelLabel.Text = "Označení otázky: " + xmlReader.GetAttribute("label");
                    }
                }
            }
        }

        public void LoadResultInfo()
        {
            ResultIdentifierLabel.Text = "Identifikátor pokusu: " + deliveryExecutionIdentifier;
            string resultTimestamp = "";
            XmlReader xmlReader = XmlReader.Create("C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + ".xml");
            while (xmlReader.Read())
            {
                if(xmlReader.Name == "testResult" && xmlReader.NodeType != XmlNodeType.EndElement)
                {
                    resultTimestamp = xmlReader.GetAttribute("datestamp");
                }
            }
            ResultTimestampLabel.Text = "Časová známka: " + resultTimestamp;

            foreach (var file in Directory.GetFiles("C:\\xampp\\exported\\testtakers"))
            {
                string extension = Path.GetExtension(file);
                if (extension == ".rdf")
                {
                    IGraph g = new Graph();
                    FileLoader.Load(g, file);
                    IEnumerable<INode> nodes = g.AllNodes;
                    int nodeLine = 1;//TODO 1: Předělat; Udělat podmínky jako if(node == ns0:userFirstName)
                    string login = "", name = "", surname = "";
                    foreach (INode node in nodes)
                    {
                        if (nodeLine == 1)
                        {
                            string[] splitByHashtag = node.ToString().Split("#");
                            if(splitByHashtag[1] != studentIdentifier)
                            {
                                break;
                            }
                        }
                        if (nodeLine == 3)
                        {
                            login = node.ToString();
                        }
                        else if (nodeLine == 9)
                        {
                            name = node.ToString();
                        }
                        else if (nodeLine == 11)
                        {
                            surname = node.ToString();
                        }
                        nodeLine++;
                        StudentNameLabel.Text = "Jméno studenta: " + name + " " + surname;
                        StudentLoginLabel.Text = "Login studenta: " + login;
                    }
                }
            }
            StudentIdentifierLabel.Text = "Identifikátor studenta: " + studentIdentifier;

            string resultsFilePath = "C:\\xampp\\exported\\results\\" + testNameIdentifier + "\\delivery_execution_" + deliveryExecutionIdentifier + "Results.txt";
            bool resultsFileExists = false;

            foreach (var file in Directory.GetFiles("C:\\xampp\\exported\\results\\" + testNameIdentifier))
            {
                if (file == resultsFilePath)
                {
                    resultsFileExists = true;
                }
            }

            if(!resultsFileExists)
            {
                string resultPointsToText = "";
                for (int i = 0; i < ItemsGridView.Rows.Count; i++)
                {
                    string itemNameIdentifier = ItemsGridView.Rows[i].Cells[2].Value.ToString();
                    string itemNumberIdentifier = ItemsGridView.Rows[i].Cells[3].Value.ToString();
                    ItemForm itemForm = new ItemForm(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier, testNumberIdentifier, false, deliveryExecutionIdentifier, studentIdentifier, false, isTeacherReviewingDeliveryResult);
                    List<double> itemPoints = itemForm.GetResultsFilePoints();
                    resultPointsToText += ItemsGridView.Rows[i].Cells[2].Value.ToString();
                    for(int j = 0; j < itemPoints.Count; j++)
                    {
                        resultPointsToText += ";" + Math.Round(itemPoints[j], 2);
                    }
                    resultPointsToText += "\n";
                }
                File.WriteAllText(resultsFilePath, resultPointsToText);
            }

            string[] resultsFileLines = File.ReadAllLines(resultsFilePath);
            double studentsPoints = 0;
            for (int i = 0; i < resultsFileLines.Length; i++)
            {
                double studentsItemPoints = 0;
                string[] splitResultsFileLineBySemicolon = resultsFileLines[i].Split(";");
                for(int j = 1; j < splitResultsFileLineBySemicolon.Length; j++)
                {
                    studentsPoints += double.Parse(splitResultsFileLineBySemicolon[j]);
                    studentsItemPoints += double.Parse(splitResultsFileLineBySemicolon[j]);
                }
                ItemsGridView.Rows[i].Cells[4].Value = studentsItemPoints + "/" + ItemsGridView.Rows[i].Cells[4].Value;
            }

            if (!testPointsDetermined)
            {
                TestPointsLabel.Text = "Počet bodů za test: N/A";
            }
            else
            {
                TestPointsLabel.Text = "Počet bodů za test: " + studentsPoints + "/" + testPoints;
            }
        }

        public string GetItemPath()
        {
            return "C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\items\\" + itemNumberIdentifier + "\\qti.xml";
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(isTeacherEditingQuestion)
            {
                new TestsForm().Show();
                Hide();
            }
            else
            {
                new ResultForm(studentIdentifier, isTeacherReviewingDeliveryResult).Show();
                Hide();
            }
        }

        private void ItemsGridView_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            chosenItem = ItemsGridView.CurrentCell.RowIndex;
            LoadItemInfo();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new ItemForm(testNameIdentifier, itemNameIdentifier, itemNumberIdentifier, testNumberIdentifier, isTeacherEditingQuestion, deliveryExecutionIdentifier, studentIdentifier, true, isTeacherReviewingDeliveryResult).Show();
            Hide();
        }
    }
}
