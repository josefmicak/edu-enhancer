
namespace TAO_Enhancer
{
    partial class TestsForm
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
            this.label1 = new System.Windows.Forms.Label();
            this.TestsGridView = new System.Windows.Forms.DataGridView();
            this.NameColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.IdentifierColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.TitleColumn = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.ManageTestButton = new System.Windows.Forms.Button();
            this.ReturnButton = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.TestsGridView)).BeginInit();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.label1.Location = new System.Drawing.Point(36, 35);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(120, 20);
            this.label1.TabIndex = 3;
            this.label1.Text = "Seznam testů";
            // 
            // TestsGridView
            // 
            this.TestsGridView.AllowUserToAddRows = false;
            this.TestsGridView.AllowUserToDeleteRows = false;
            this.TestsGridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.TestsGridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.NameColumn,
            this.IdentifierColumn,
            this.TitleColumn});
            this.TestsGridView.Location = new System.Drawing.Point(36, 80);
            this.TestsGridView.Name = "TestsGridView";
            this.TestsGridView.ReadOnly = true;
            this.TestsGridView.RowTemplate.Height = 25;
            this.TestsGridView.Size = new System.Drawing.Size(506, 150);
            this.TestsGridView.TabIndex = 0;
            this.TestsGridView.CellClick += new System.Windows.Forms.DataGridViewCellEventHandler(this.TestsGridView_CellClick);
            this.TestsGridView.SelectionChanged += new System.EventHandler(this.TestsGridView_SelectionChanged);
            // 
            // NameColumn
            // 
            this.NameColumn.HeaderText = "Jméno ve složce";
            this.NameColumn.Name = "NameColumn";
            this.NameColumn.ReadOnly = true;
            this.NameColumn.Width = 150;
            // 
            // IdentifierColumn
            // 
            this.IdentifierColumn.HeaderText = "Identifikátor";
            this.IdentifierColumn.Name = "IdentifierColumn";
            this.IdentifierColumn.ReadOnly = true;
            this.IdentifierColumn.Width = 150;
            // 
            // TitleColumn
            // 
            this.TitleColumn.HeaderText = "Nadpis";
            this.TitleColumn.Name = "TitleColumn";
            this.TitleColumn.ReadOnly = true;
            this.TitleColumn.Width = 150;
            // 
            // ManageTestButton
            // 
            this.ManageTestButton.Location = new System.Drawing.Point(36, 263);
            this.ManageTestButton.Name = "ManageTestButton";
            this.ManageTestButton.Size = new System.Drawing.Size(98, 23);
            this.ManageTestButton.TabIndex = 4;
            this.ManageTestButton.Text = "Správa testu";
            this.ManageTestButton.UseVisualStyleBackColor = true;
            this.ManageTestButton.Click += new System.EventHandler(this.ManageTestButton_Click);
            // 
            // ReturnButton
            // 
            this.ReturnButton.Location = new System.Drawing.Point(231, 380);
            this.ReturnButton.Name = "ReturnButton";
            this.ReturnButton.Size = new System.Drawing.Size(75, 23);
            this.ReturnButton.TabIndex = 5;
            this.ReturnButton.Text = "Zpět";
            this.ReturnButton.UseVisualStyleBackColor = true;
            this.ReturnButton.Click += new System.EventHandler(this.ReturnButton_Click);
            // 
            // TestsForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(564, 450);
            this.Controls.Add(this.ReturnButton);
            this.Controls.Add(this.ManageTestButton);
            this.Controls.Add(this.TestsGridView);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "TestsForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "TestsForm";
            ((System.ComponentModel.ISupportInitialize)(this.TestsGridView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.DataGridView TestsGridView;
        private System.Windows.Forms.Button ManageTestButton;
        private System.Windows.Forms.Button ReturnButton;
        private System.Windows.Forms.DataGridViewTextBoxColumn NameColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn IdentifierColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn TitleColumn;
    }
}