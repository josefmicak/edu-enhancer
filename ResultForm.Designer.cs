
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
            this.button1 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.ResultsGridView)).BeginInit();
            this.SuspendLayout();
            // 
            // ResultsGridView
            // 
            this.ResultsGridView.AllowUserToAddRows = false;
            this.ResultsGridView.AllowUserToDeleteRows = false;
            this.ResultsGridView.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.ResultsGridView.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.TestNameIdentifierColumn,
            this.TimestampColumn});
            this.ResultsGridView.Location = new System.Drawing.Point(22, 82);
            this.ResultsGridView.Name = "ResultsGridView";
            this.ResultsGridView.ReadOnly = true;
            this.ResultsGridView.RowTemplate.Height = 25;
            this.ResultsGridView.Size = new System.Drawing.Size(359, 150);
            this.ResultsGridView.TabIndex = 0;
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
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(353, 406);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 1;
            this.button1.Text = "Zpět";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point);
            this.label1.Location = new System.Drawing.Point(22, 46);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(147, 20);
            this.label1.TabIndex = 2;
            this.label1.Text = "Seznam výsledků";
            // 
            // ResultForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.ResultsGridView);
            this.Name = "ResultForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "ResultForm";
            ((System.ComponentModel.ISupportInitialize)(this.ResultsGridView)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.DataGridView ResultsGridView;
        private System.Windows.Forms.DataGridViewTextBoxColumn TestNameIdentifierColumn;
        private System.Windows.Forms.DataGridViewTextBoxColumn TimestampColumn;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Label label1;
    }
}