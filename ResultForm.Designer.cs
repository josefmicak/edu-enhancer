
namespace TAO_Enhancer
{
    partial class ResultForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.ResultsGridView = new System.Windows.Forms.DataGridView();
            this.TestNameIdentifierColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.TimestampColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.AttemptIdentifier = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.StudentLoginGB = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.ReturnButton = new System.Windows.Forms.Button();
            this.ShowTestButton = new System.Windows.Forms.Button();
            this.ResultListGB = new System.Windows.Forms.GroupBox();
            this.StudentGB = new System.Windows.Forms.GroupBox();
            this.StudentEmailLabel = new System.Windows.Forms.Label();
            this.StudentLoginLabel = new System.Windows.Forms.Label();
            this.StudentNameLabel = new System.Windows.Forms.Label();
            this.AmountOfTestsLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.ResultsGridView)).BeginInit();
            this.ResultListGB.SuspendLayout();
            this.StudentGB.SuspendLayout();
            this.SuspendLayout();
            // 
            // ResultsGridView
            // 
            this.ResultsGridView.AllowUserToAddRows = false;
            this.ResultsGridView.AllowUserToDeleteRows = false;
            this.ResultsGridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.ResultsGridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.TestNameIdentifierColumn,
            this.TimestampColumn,
            this.AttemptIdentifier,
            this.StudentLoginGB});
            this.ResultsGridView.Location = new System.Drawing.Point(58, 70);
            this.ResultsGridView.Name = "ResultsGridView";
            this.ResultsGridView.ReadOnly = true;
            this.ResultsGridView.RowTemplate.Height = 25;
            this.ResultsGridView.Size = new System.Drawing.Size(613, 282);
            this.ResultsGridView.TabIndex = 0;
            this.ResultsGridView.CellClick += new System.Windows.Forms.DataGridViewCellEventHandler(this.ResultsGridView_CellClick);
            this.ResultsGridView.SelectionChanged += new System.EventHandler(this.ResultsGridView_SelectionChanged);
            // 
            // TestNameIdentifierColumn
            // 
            this.TestNameIdentifierColumn.HeaderText = "Jmenný identifikátor testu";
            this.TestNameIdentifierColumn.Name = "TestNameIdentifierColumn";
            this.TestNameIdentifierColumn.ReadOnly = true;
            this.TestNameIdentifierColumn.Width = 150;
            // 
            // TimestampColumn
            // 
            this.TimestampColumn.HeaderText = "Časová známka";
            this.TimestampColumn.Name = "TimestampColumn";
            this.TimestampColumn.ReadOnly = true;
            this.TimestampColumn.Width = 150;
            // 
            // AttemptIdentifier
            // 
            this.AttemptIdentifier.HeaderText = "Identifikátor pokusu";
            this.AttemptIdentifier.Name = "AttemptIdentifier";
            this.AttemptIdentifier.ReadOnly = true;
            this.AttemptIdentifier.Width = 150;
            // 
            // StudentLoginGB
            // 
            this.StudentLoginGB.HeaderText = "Login studenta";
            this.StudentLoginGB.Name = "StudentLoginGB";
            this.StudentLoginGB.ReadOnly = true;
            // 
            // ReturnButton
            // 
            this.ReturnButton.Location = new System.Drawing.Point(539, 506);
            this.ReturnButton.Name = "ReturnButton";
            this.ReturnButton.Size = new System.Drawing.Size(75, 23);
            this.ReturnButton.TabIndex = 1;
            this.ReturnButton.Text = "Zpět";
            this.ReturnButton.UseVisualStyleBackColor = true;
            this.ReturnButton.Click += new System.EventHandler(this.ReturnButton_Click);
            // 
            // ShowTestButton
            // 
            this.ShowTestButton.Location = new System.Drawing.Point(328, 365);
            this.ShowTestButton.Name = "ShowTestButton";
            this.ShowTestButton.Size = new System.Drawing.Size(90, 23);
            this.ShowTestButton.TabIndex = 3;
            this.ShowTestButton.Text = "Zobrazit test";
            this.ShowTestButton.UseVisualStyleBackColor = true;
            this.ShowTestButton.Click += new System.EventHandler(this.ShowTestButton_Click);
            // 
            // ResultListGB
            // 
            this.ResultListGB.Controls.Add(this.ResultsGridView);
            this.ResultListGB.Controls.Add(this.ShowTestButton);
            this.ResultListGB.Location = new System.Drawing.Point(397, 44);
            this.ResultListGB.Name = "ResultListGB";
            this.ResultListGB.Size = new System.Drawing.Size(740, 404);
            this.ResultListGB.TabIndex = 4;
            this.ResultListGB.TabStop = false;
            this.ResultListGB.Text = "Seznam výsledků";
            // 
            // StudentGB
            // 
            this.StudentGB.Controls.Add(this.StudentEmailLabel);
            this.StudentGB.Controls.Add(this.StudentLoginLabel);
            this.StudentGB.Controls.Add(this.StudentNameLabel);
            this.StudentGB.Location = new System.Drawing.Point(13, 44);
            this.StudentGB.Name = "StudentGB";
            this.StudentGB.Size = new System.Drawing.Size(357, 250);
            this.StudentGB.TabIndex = 5;
            this.StudentGB.TabStop = false;
            this.StudentGB.Text = "Student";
            // 
            // StudentEmailLabel
            // 
            this.StudentEmailLabel.AutoSize = true;
            this.StudentEmailLabel.Location = new System.Drawing.Point(7, 109);
            this.StudentEmailLabel.Name = "StudentEmailLabel";
            this.StudentEmailLabel.Size = new System.Drawing.Size(105, 15);
            this.StudentEmailLabel.TabIndex = 2;
            this.StudentEmailLabel.Text = "StudentEmailLabel";
            // 
            // StudentLoginLabel
            // 
            this.StudentLoginLabel.AutoSize = true;
            this.StudentLoginLabel.Location = new System.Drawing.Point(7, 70);
            this.StudentLoginLabel.Name = "StudentLoginLabel";
            this.StudentLoginLabel.Size = new System.Drawing.Size(106, 15);
            this.StudentLoginLabel.TabIndex = 1;
            this.StudentLoginLabel.Text = "StudentLoginLabel";
            // 
            // StudentNameLabel
            // 
            this.StudentNameLabel.AutoSize = true;
            this.StudentNameLabel.Location = new System.Drawing.Point(7, 32);
            this.StudentNameLabel.Name = "StudentNameLabel";
            this.StudentNameLabel.Size = new System.Drawing.Size(108, 15);
            this.StudentNameLabel.TabIndex = 0;
            this.StudentNameLabel.Text = "StudentNameLabel";
            // 
            // AmountOfTestsLabel
            // 
            this.AmountOfTestsLabel.AutoSize = true;
            this.AmountOfTestsLabel.Location = new System.Drawing.Point(13, 322);
            this.AmountOfTestsLabel.Name = "AmountOfTestsLabel";
            this.AmountOfTestsLabel.Size = new System.Drawing.Size(117, 15);
            this.AmountOfTestsLabel.TabIndex = 3;
            this.AmountOfTestsLabel.Text = "AmountOfTestsLabel";
            // 
            // ResultForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1256, 565);
            this.Controls.Add(this.AmountOfTestsLabel);
            this.Controls.Add(this.StudentGB);
            this.Controls.Add(this.ResultListGB);
            this.Controls.Add(this.ReturnButton);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "ResultForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "ResultForm";
            ((System.ComponentModel.ISupportInitialize)(this.ResultsGridView)).EndInit();
            this.ResultListGB.ResumeLayout(false);
            this.StudentGB.ResumeLayout(false);
            this.StudentGB.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView ResultsGridView;
        private System.Windows.Forms.Button ReturnButton;
        private System.Windows.Forms.Button ShowTestButton;
        private System.Windows.Forms.DataGridViewTextBoxColumn TestNameIdentifierColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn TimestampColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn AttemptIdentifier;
        private System.Windows.Forms.GroupBox ResultListGB;
        private System.Windows.Forms.GroupBox StudentGB;
        private System.Windows.Forms.Label StudentLoginLabel;
        private System.Windows.Forms.Label StudentNameLabel;
        private System.Windows.Forms.Label StudentEmailLabel;
        private System.Windows.Forms.Label AmountOfTestsLabel;
        private System.Windows.Forms.DataGridViewTextBoxColumn StudentLoginGB;
    }
}